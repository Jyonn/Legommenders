import os
from typing import cast

import numpy as np
import torch
from peft import get_peft_model, LoraConfig
from pigmento import pnt
from torch import nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from loader.data_set import DataSet
from loader.env import Env
from loader.pager.lm_layer_pager import LMLayerPager
from model.common.attention import AdditiveAttention
from model.inputer.concat_inputer import ConcatInputer
from model.operators.ada_operator import AdaOperatorConfig
from model.operators.lm_operator import LMOperator
from utils import bars
from utils.config_init import ModelInit


class OnceOperatorConfig(AdaOperatorConfig):
    def __init__(
            self,
            tune_from: int | None = 0,  # last frozen layer; 0 and None disable caching
            use_lora=True,
            lora_alpha=128,
            lora_r=32,
            lora_dropout=0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.tune_from = tune_from
        self.use_lora = use_lora
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout


class OnceOperator(LMOperator):
    config_class = OnceOperatorConfig
    inputer_class = ConcatInputer
    inputer: ConcatInputer
    config: OnceOperatorConfig
    dtype = torch.float32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = AutoModel.from_pretrained(
            self.transformer_key, trust_remote_code=True, dtype=self.dtype,
        )
        self.num_hidden_layers = self.get_layer_nums()
        os.makedirs(self._cache_base_dir, exist_ok=True)

        if self.config.tune_from and self.config.tune_from < 0:
            self.config.tune_from = self.num_hidden_layers + self.config.tune_from
        if self.config.tune_from is not None and not 0 <= self.config.tune_from < self.num_hidden_layers:
            raise ValueError(f'tune_from must select a layer from 0 to {self.num_hidden_layers - 1}')

        self.hidden_weights = None  # [N, L, D]
        self.attention_mask = None  # [N, L]

        self.linear = nn.Linear(self.config.input_dim, self.config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.additive_hidden_size,
        )

    def use_lm_cache(self):
        return bool(self.config.tune_from)

    @property
    def transformer_key(self):
        return ModelInit.get(self.operator_name)

    def get_layer_nums(self):
        return self.transformer.config.num_hidden_layers

    def __str__(self):
        return self.operator_name

    @property
    def _cache_base_dir(self):
        return os.path.join('cache', Env.ph.data_name, self.operator_name, 'layer-outputs')

    def _cache_exists(self, layer):
        return os.path.exists(self._get_cache_path(layer))

    def _get_cache_path(self, layer):
        return os.path.join(self._cache_base_dir, f'layer_{layer}.npy')

    def _get_mask_path(self):
        return os.path.join(self._cache_base_dir, 'mask.npy')

    def _slice_transformer_layers(self):
        raise NotImplementedError

    def _lora_encoder(self, peft_config):
        self.transformer = get_peft_model(cast(PreTrainedModel, self.transformer), peft_config)
        self.transformer.print_trainable_parameters()

    def _load_hidden_states(self):
        if self._cache_exists(self.config.tune_from):
            pnt(f'loading cached hidden states from {self._get_cache_path(self.config.tune_from)}')
        else:
            pnt(f'caching item hidden states from layer {self.config.tune_from} at runtime')
            pnt('to accelerate this process, please consider to use splitter.py to cache all required layers at once')
            self.cache([self.config.tune_from])
        hidden_weights = np.load(self._get_cache_path(self.config.tune_from))
        self.hidden_weights = torch.from_numpy(hidden_weights)
        attention_mask = np.load(self._get_mask_path())
        self.attention_mask = torch.from_numpy(attention_mask)
        self.hidden_weights = self.hidden_weights.view(*self.attention_mask.shape[:2], self.hidden_weights.shape[-1])
        pnt(f'hidden_weights.shape: {self.hidden_weights.shape}')  # [N, L, D]
        pnt(f'attention_mask.shape: {self.attention_mask.shape}')  # [N, L]

        nan_mask = torch.isnan(self.hidden_weights).any(dim=-1)  # [N, L]
        self.hidden_weights[nan_mask] = torch.rand_like(self.hidden_weights[nan_mask]).to(self.hidden_weights.dtype)
        nan_mask_ = nan_mask.any(dim=-1)  # [N]
        template_mask = torch.zeros_like(self.attention_mask[0], dtype=self.attention_mask.dtype)
        template_mask[0] = 1  # Set the first position to 1
        self.attention_mask[nan_mask_] = template_mask

    def _prepare_network(self):
        if self.use_lm_cache():
            self._load_hidden_states()
            self._slice_transformer_layers()
            torch.cuda.empty_cache()
            pnt(f'sliced transformer layers, '
                f'{self.num_hidden_layers} -> {self.num_hidden_layers - self.config.tune_from - 1}')

        if ((not self.use_lm_cache()
                or self.config.tune_from < self.num_hidden_layers - 1)
                and self.config.use_lora):
            if not isinstance(self.config.lora_r, int):
                raise ValueError('lora_r should be an integer')
            if not isinstance(self.config.lora_alpha, int):
                raise ValueError('lora_alpha should be an integer')
            if not isinstance(self.config.lora_dropout, float):
                raise ValueError('lora_dropout should be a float')
            peft_config = LoraConfig(
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout
            )
            self._lora_encoder(peft_config)

    def get_pretrained_parameter_names(self):
        return ['transformer']

    def _forward(
            self,
            inputs_embeds,
            attention_mask,
            output_hidden_states: bool = True,  # must be True, for caching hidden states
    ):
        output: BaseModelOutputWithPast = self.transformer(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if output_hidden_states:
            # Use the final output for the last layer, including norm/projection.
            return (*output.hidden_states[1:-1], output.last_hidden_state)
        return output.last_hidden_state

    def forward(self, embeddings, mask=None, **kwargs):
        if not self.use_lm_cache():
            mask = mask.to(Env.device)
            outputs = self._forward(
                inputs_embeds=embeddings,
                attention_mask=mask,
                output_hidden_states=False,
            )
        else:
            indices = embeddings.cpu()  # [B]
            hidden_states = self.hidden_weights[indices].to(self.dtype).to(Env.device)  # [B, L, D]
            mask = self.attention_mask[indices].to(Env.device)  # [B, L]
            if self.config.tune_from == self.num_hidden_layers - 1:
                outputs = hidden_states
            else:
                outputs = self._loop_forward(hidden_states=hidden_states, attention_mask=mask)

        outputs = outputs.to(torch.float32)

        outputs = self.linear(outputs)  # [B, L, D]
        outputs = self.additive_attention(outputs, mask)  # [B, D]

        # nan_mask = torch.isnan(outputs).any(dim=-1)
        # outputs[nan_mask] = self.nan_embedding

        return outputs

    def _loop_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        raise NotImplementedError

    @property
    def cache_hidden_size(self):
        return self.config.input_dim

    def get_cache_hidden_size(self, layer):
        return self.cache_hidden_size

    @torch.no_grad()
    def cache(self, layers):
        if any(not 0 <= layer < self.num_hidden_layers for layer in layers):
            raise ValueError(f'Cache layers must be between 0 and {self.num_hidden_layers - 1}')

        dataset = DataSet(ut=self.lego_config.item_ut)
        contents = []
        for sample in bars.DescBar(desc='Caching Item Content')(dataset):
            contents.append(self.inputer(sample))

        self.to(Env.device)

        pager = LMLayerPager(
            inputer=self.inputer,
            layers=layers,
            hidden_sizes=[self.get_cache_hidden_size(layer) for layer in layers],
            contents=contents,
            model=self._forward,
            page_size=self.lego_config.item_page_size,
        )

        was_training = self.transformer.training
        self.transformer.eval()
        try:
            pager.run()
        finally:
            self.transformer.train(was_training)

        for index, layer in enumerate(layers):
            np.save(self._get_cache_path(layer), pager.final_features[index].float().numpy())
        np.save(self._get_mask_path(), pager.final_masks.numpy())

        pnt(f'caches saved to {self._cache_base_dir}')


