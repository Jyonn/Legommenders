import os

import numpy as np
import torch
from peft import get_peft_model, LoraConfig
from pigmento import pnt
from torch import nn
from tqdm import tqdm
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from loader.data_set import DataSet
from loader.env import Env
from loader.pager.llm_split_pager import LLMSplitPager
from model.common.attention import AdditiveAttention
from model.inputer.concat_inputer import ConcatInputer
from model.operators.attention_operator import AttentionOperatorConfig
from model.operators.base_operator import BaseOperator
from utils.config_init import ModelInit


class BaseLMOperatorConfig(AttentionOperatorConfig):
    def __init__(
            self,
            tune_from: int = 0,  # number of layer, such as 0, 30, 31, etc.
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


class BaseLMOperator(BaseOperator):
    config_class = BaseLMOperatorConfig
    inputer_class = ConcatInputer
    inputer: ConcatInputer
    config: BaseLMOperatorConfig
    dtype = torch.float32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = AutoModel.from_pretrained(ModelInit.get(str(self)), trust_remote_code=True)
        self.num_hidden_layers = self.get_layer_nums()
        os.makedirs(self._cache_base_dir, exist_ok=True)

        if self.config.tune_from < 0:
            self.config.tune_from = self.num_hidden_layers + self.config.tune_from

        self.hidden_weights = None  # [N, L, D]
        self.attention_mask = None  # [N, L, L]

        self.linear = nn.Linear(self.config.input_dim, self.config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.additive_hidden_size,
        )

    @property
    def operator_name(self):
        return self.__class__.__name__.replace('Operator', '').lower()

    def get_layer_nums(self):
        return self.transformer.config.num_hidden_layers

    def __str__(self):
        return self.operator_name

    @property
    def _cache_base_dir(self):
        return os.path.join('cache', Env.path_hub.data_name, self.operator_name)

    def _cache_exists(self, layer):
        return os.path.exists(self._get_cache_path(layer))

    def _get_cache_path(self, layer):
        return os.path.join(self._cache_base_dir, f'layer_{layer}.npy')

    def _get_mask_path(self):
        return os.path.join(self._cache_base_dir, 'mask.npy')

    def _slice_transformer_layers(self):
        raise NotImplementedError

    def _lora_encoder(self, peft_config):
        self.transformer = get_peft_model(self.transformer, peft_config)
        self.transformer.print_trainable_parameters()

    def _load_hidden_states(self):
        if not self._cache_exists(self.config.tune_from):
            if self.config.tune_from >= self.num_hidden_layers:
                raise ValueError(f'tune_from should be less than {self.num_hidden_layers}')
            pnt(f'caching item hidden states from layer {self.config.tune_from} at runtime')
            pnt('to accelerate this process, please consider to use splitter.py to cache all required layers at once')
            self.cache([self.config.tune_from])
        hidden_weights = np.load(self._get_cache_path(self.config.tune_from))
        self.hidden_weights = torch.from_numpy(hidden_weights).to(Env.device)
        attention_mask = np.load(self._get_mask_path())
        self.attention_mask = torch.from_numpy(attention_mask).to(Env.device)
        self.hidden_weights = self.hidden_weights.view(*self.attention_mask.shape[:2], self.hidden_weights.shape[-1])
        pnt(f'hidden_weights.shape: {self.hidden_weights.shape}')
        pnt(f'attention_mask.shape: {self.attention_mask.shape}')

        self.transformer = self.transformer.to(self.dtype)
        pnt(f'switched transformer dtype to {self.dtype}')

    def _prepare_network(self):
        if self.config.tune_from:
            self._load_hidden_states()
            self._slice_transformer_layers()
            pnt(f'sliced transformer layers, '
                f'{self.num_hidden_layers} -> {self.num_hidden_layers - self.config.tune_from - 1}')

        if self.config.tune_from < self.num_hidden_layers - 1 and self.config.use_lora:
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if output_hidden_states:
            return output.hidden_states
        return output.last_hidden_state

    def forward(self, embeddings, mask=None, **kwargs):
        if not self.config.tune_from:
            inputs_embeds = embeddings.to(self.dtype)
            mask = mask.to(Env.device).to(self.dtype)
            outputs = self._forward(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                output_hidden_states=False,
            )
        else:
            indices = embeddings  # [B]
            hidden_states = self.hidden_weights[indices].to(self.dtype)  # [B, L, D]
            mask = self.attention_mask[indices].to(self.dtype)  # [B, L, L]
            outputs = self._loop_forward(
                hidden_states=hidden_states,
                attention_mask=mask,
            )  # [B, L, D]

        outputs = outputs.to(torch.float32)

        outputs = self.linear(outputs)  # [B, L, D]
        outputs = self.additive_attention(outputs, mask)  # [B, D]

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

    def cache(self, layers):
        dataset = DataSet(ut=self.lego_config.item_ut)
        contents = []
        for sample in tqdm(dataset):
            contents.append(self.inputer(sample))

        self.to(Env.device)

        pager = LLMSplitPager(
            inputer=self.inputer,
            layers=layers,
            hidden_size=self.cache_hidden_size,
            contents=contents,
            model=self._forward,
            page_size=self.lego_config.item_page_size,
        )

        pager.run()

        for index, layer in enumerate(layers):
            np.save(self._get_cache_path(layer), pager.final_features[index].float().numpy())
        np.save(self._get_mask_path(), pager.final_masks.numpy())

        pnt(f'caches saved to {self._cache_base_dir}')


