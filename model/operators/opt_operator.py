import abc

import torch
from transformers import OPTModel

from model.operators.lm_operator import BaseLMOperator


class OPTOperator(BaseLMOperator, abc.ABC):
    dtype = torch.bfloat16
    transformer: OPTModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer.decoder.embed_tokens = None

        self._prepare_network()

    def _slice_transformer_layers(self):
        self.transformer.decoder.layers = self.transformer.decoder.layers[self.config.tune_from + 1:]

    def _loop_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.Tensor,
    ):
        opt = self.transformer

        input_shape = hidden_states.size()[:-1]

        causal_attention_mask, attention_mask = opt.decoder._update_causal_mask(
            hidden_states, input_shape, 0, attention_mask, None, None
        )

        position_ids = torch.cumsum(attention_mask, dim=1)
        position_ids = (position_ids * attention_mask - 1).long()

        for decoder_layer in opt.decoder.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                position_ids=position_ids,
                layer_head_mask=None,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
            )

            hidden_states = layer_outputs[0]

        if opt.decoder.final_layer_norm is not None:
            hidden_states = opt.decoder.final_layer_norm(hidden_states)

        if opt.decoder.project_out is not None:
            hidden_states = opt.decoder.project_out(hidden_states)

        return hidden_states

    @property
    def cache_hidden_size(self):
        return self.transformer.config.hidden_size


class OPTBaseOperator(OPTOperator):
    pass


class OPTLargeOperator(OPTOperator):
    pass
