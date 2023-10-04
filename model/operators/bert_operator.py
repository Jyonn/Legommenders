from typing import Optional

import torch
from peft import get_peft_model
from transformers import BertModel

from model.inputer.llm_concat_inputer import BertConcatInputer
from model.operators.base_llm_operator import BaseLLMOperator


class BertOperator(BaseLLMOperator):
    inputer_class = BertConcatInputer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = BertModel.from_pretrained(self.config.llm_dir)  # type: BertModel
        self.transformer.embeddings.word_embeddings = None
        self.layer_split(self.transformer.config.num_hidden_layers)

    def _slice_transformer_layers(self):
        self.transformer.encoder.layer = self.transformer.encoder.layer[self.config.layer_split + 1:]

    def _lora_encoder(self, peft_config):
        self.transformer.encoder = get_peft_model(self.transformer.encoder, peft_config)
        self.transformer.encoder.print_trainable_parameters()

    def get_all_hidden_states(
            self,
            hidden_states,
            attention_mask,
    ):
        bert = self.transformer

        input_shape = hidden_states.size()[:-1]
        device = hidden_states.device

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        hidden_states = bert.embeddings(
            token_type_ids=token_type_ids,
            inputs_embeds=hidden_states,
        )

        return self._layer_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    def _layer_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        bert = self.transformer

        attention_mask = bert.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1])
        all_hidden_states = ()

        for i, layer_module in enumerate(bert.encoder.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states

    def layer_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        return self._layer_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )[0]
