import os
from typing import Optional

import numpy as np
import torch
from peft import get_peft_model, LoraConfig
from pigmento import pnt
from torch import nn
from transformers import PreTrainedModel

from loader.meta import Meta
from model.common.attention import AdditiveAttention
from model.inputer.natural_concat_inputer import NaturalConcatInputer
from model.operators.attention_operator import AttentionOperatorConfig
from model.operators.base_operator import BaseOperator


class BaseLLMOperatorConfig(AttentionOperatorConfig):
    def __init__(
            self,
            llm_dir: str,
            layer_split: int = 0,  # [0, 24, 28, 30, 31]
            weights_dir: Optional[str] = None,
            lora=True,
            lora_alpha=128,
            lora_r=32,
            lora_dropout=0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_dir = llm_dir
        self.layer_split = layer_split
        self.weights_dir = weights_dir
        self.lora = lora
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout


class BaseLLMOperator(BaseOperator):
    config_class = BaseLLMOperatorConfig
    inputer_class = NaturalConcatInputer
    inputer: NaturalConcatInputer
    config: BaseLLMOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = None  # type: Optional[PreTrainedModel]

        self.hidden_weights = None
        self.attention_mask = None

        self.linear = nn.Linear(self.config.embed_hidden_size, self.config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
        )

    def _slice_transformer_layers(self):
        raise NotImplementedError

    def _lora_encoder(self, peft_config):
        self.transformer = get_peft_model(self.transformer, peft_config)
        self.transformer.print_trainable_parameters()

    def layer_split(self, num_hidden_layers):
        if self.config.layer_split:
            self._slice_transformer_layers()
            # self.transformer.layers = self.transformer.layers[self.config.layer_split+1:]
            hidden_weights = np.load(os.path.join(self.config.weights_dir, f'layer_{self.config.layer_split}.npy'))
            self.hidden_weights = torch.from_numpy(hidden_weights).to(Meta.device)
            attention_mask = np.load(os.path.join(self.config.weights_dir, 'mask.npy'))
            self.attention_mask = torch.from_numpy(attention_mask).to(Meta.device)
            self.hidden_weights = self.hidden_weights.view(*self.attention_mask.shape[:2], self.hidden_weights.shape[-1])
            pnt(f'hidden_weights.shape: {self.hidden_weights.shape}')
            pnt(f'attention_mask.shape: {self.attention_mask.shape}')

        if self.config.layer_split < num_hidden_layers - 1 and self.config.lora:
            peft_config = LoraConfig(
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout
            )
            self._lora_encoder(peft_config)

    def get_pretrained_parameter_names(self):
        return ['transformer']

    def get_all_hidden_states(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.Tensor,
    ):
        raise NotImplementedError

    def forward(self, embeddings, mask=None, **kwargs):
        if not self.config.layer_split:
            mask = mask.to(Meta.device)

            llm_output = self.transformer(
                inputs_embeds=embeddings,
                attention_mask=mask,
                return_dict=True,
            )
            outputs = llm_output.last_hidden_state  # [B, L, D]
        else:
            indices = embeddings  # [B]
            hidden_states = self.hidden_weights[indices]  # [B, L, D]
            mask = self.attention_mask[indices]  # [B, L, L]
            outputs = self.layer_forward(
                hidden_states,
                mask,
            )  # [B, L, D]

        outputs = self.linear(outputs)  # [B, L, D]
        outputs = self.additive_attention(outputs, mask)  # [B, D]

        return outputs

    def layer_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        return self.get_all_hidden_states(hidden_states, attention_mask)[-1]
