import abc

import torch
from transformers import LlamaModel

from model.operators.lm_operator import BaseLMOperator


class LlamaOperator(BaseLMOperator, abc.ABC):
    dtype = torch.bfloat16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer: LlamaModel
        self.transformer.embed_tokens = None

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

        self._prepare_network()

    def _slice_transformer_layers(self):
        self.transformer.layers = self.transformer.layers[self.config.tune_from + 1:]

    def _loop_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.Tensor,
    ):
        llama = self.transformer

        cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)

        causal_mask = llama._update_causal_mask(
            attention_mask, hidden_states, cache_position, None, False
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = llama.rotary_emb(hidden_states, position_ids)

        for decoder_layer in llama.layers:
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        hidden_states = llama.norm(hidden_states)

        return hidden_states


class Llama1Operator(LlamaOperator):
    pass


class Llama2Operator(LlamaOperator):
    pass


class Llama3Operator(LlamaOperator):
    pass
