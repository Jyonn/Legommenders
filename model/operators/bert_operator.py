import abc

import torch
from peft import get_peft_model, PeftConfig
from transformers import BertModel

from model.operators.lm_operator import BaseLMOperator


class BertOperator(BaseLMOperator, abc.ABC):
    transformer: BertModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer.embeddings.word_embeddings = None
        self._prepare_network()

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

    def _slice_transformer_layers(self):
        self.transformer.encoder.layer = self.transformer.encoder.layer[self.config.tune_from + 1:]

    def _lora_encoder(self, peft_config: PeftConfig):
        self.transformer.encoder = get_peft_model(self.transformer.encoder, peft_config)
        self.transformer.encoder.print_trainable_parameters()

    def _loop_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        input_shape = hidden_states.size()[:-1]
        extended_attention_mask = self.transformer.get_extended_attention_mask(attention_mask, input_shape)

        output = self.transformer.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state


class BertBaseOperator(BertOperator):
    pass


class BertLargeOperator(BertOperator):
    pass
