import os
from typing import Optional

import numpy as np
import torch
from peft import get_peft_model, LoraConfig
from torch import nn
from transformers import LlamaModel

from loader.global_setting import Setting
from model.common.attention import AdditiveAttention
from model.inputer.natural_concat_inputer import NaturalConcatInputer
from model.operator.attention_operator import AttentionOperatorConfig
from model.operator.base_operator import BaseOperator


class LlamaOperatorConfig(AttentionOperatorConfig):
    def __init__(
            self,
            llama_dir: str,
            layer_split: int = 0,  # [0, 24, 28, 30, 31]
            weights_dir: Optional[str] = None,
            lora = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.llama_dir = llama_dir
        self.layer_split = layer_split
        self.weights_dir = weights_dir
        self.lora = lora


class LlamaOperator(BaseOperator):
    config_class = LlamaOperatorConfig
    inputer_class = NaturalConcatInputer
    config: LlamaOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = LlamaModel.from_pretrained(self.config.llama_dir)  # type: LlamaModel
        self.transformer.embed_tokens = None

        self.hidden_weights = None
        self.attention_mask = None
        if self.config.layer_split:
            self.transformer.layers = self.transformer.layers[self.config.layer_split+1:]
            hidden_weights = np.load(os.path.join(self.config.weights_dir, f'layer_{self.config.layer_split}.npy'))
            self.hidden_weights = torch.from_numpy(hidden_weights).to(Setting.device)
            attention_mask = np.load(os.path.join(self.config.weights_dir, 'mask.npy'))
            self.attention_mask = torch.from_numpy(attention_mask).to(Setting.device)
            self.hidden_weights = self.hidden_weights.view(*self.attention_mask.shape[:2], self.hidden_weights.shape[-1])
            self.print(f'hidden_weights.shape: {self.hidden_weights.shape}')
            self.print(f'attention_mask.shape: {self.attention_mask.shape}')

        if self.config.layer_split < self.transformer.config.num_hidden_layers - 1 and self.config.lora:
            peft_config = LoraConfig(
                inference_mode=False, r=32, lora_alpha=128, lora_dropout=0.1)
            self.transformer = get_peft_model(self.transformer, peft_config)
            self.transformer.print_trainable_parameters()

        # for param in self.transformer.layers[:self.config.freeze_layers].parameters():
        #     param.requires_grad = False

        self.linear = nn.Linear(self.transformer.config.hidden_size, self.config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
        )

    def _get_all_hidden_states(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.Tensor,
    ):
        llama = self.transformer  # type: LlamaModel

        batch_size, seq_length, _ = hidden_states.shape

        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        attention_mask = llama._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )
        # hidden_states = hidden_states

        # decoder layers
        all_hidden_states = ()

        for idx, decoder_layer in enumerate(llama.layers):
            all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = llama.norm(hidden_states)

        # add hidden states from the last decoder layer
        all_hidden_states += (hidden_states,)

        return all_hidden_states, attention_mask

    def get_all_hidden_states(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Setting.device)
        all_hidden_states = self._get_all_hidden_states(
            hidden_states=embeddings,
            attention_mask=mask,
        )
        return all_hidden_states

    def layer_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        llama = self.transformer  # type: LlamaModel

        batch_size, seq_length, _ = hidden_states.shape

        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        attention_mask = llama._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        for idx, decoder_layer in enumerate(llama.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )

            hidden_states = layer_outputs[0]

        if llama.layers:
            hidden_states = llama.norm(hidden_states)

        return hidden_states

    def forward(self, embeddings, mask=None, **kwargs):
        if not self.config.layer_split:
            mask = mask.to(Setting.device)

            transformer_output = self.transformer(
                inputs_embeds=embeddings,
                attention_mask=mask,
                return_dict=True,
            )
            outputs = transformer_output.last_hidden_state  # [B, L, D]
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
