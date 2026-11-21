import abc

import torch
from transformers import OPTModel

from model.operators.once_operator import OnceOperator


class OPTOperator(OnceOperator, abc.ABC):
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

        seq_len = hidden_states.size(1)
        positions = torch.arange(seq_len, device=hidden_states.device)
        future = positions[None, :] > positions[:, None]
        blocked = future | ~attention_mask[:, None, None, :].bool()
        causal_mask = torch.zeros_like(blocked, dtype=hidden_states.dtype)
        causal_mask = causal_mask.masked_fill(blocked, torch.finfo(hidden_states.dtype).min)

        for decoder_layer in opt.decoder.layers:
            output = decoder_layer(hidden_states, attention_mask=causal_mask, use_cache=False)
            hidden_states = output[0] if isinstance(output, tuple) else output

        if opt.decoder.final_layer_norm is not None:
            hidden_states = opt.decoder.final_layer_norm(hidden_states)

        if opt.decoder.project_out is not None:
            hidden_states = opt.decoder.project_out(hidden_states)

        return hidden_states

    @property
    def cache_hidden_size(self):
        return self.transformer.config.hidden_size

    def get_cache_hidden_size(self, layer):
        if layer == self.num_hidden_layers - 1:
            return self.transformer.config.word_embed_proj_dim
        return self.cache_hidden_size


class OPTBaseOperator(OPTOperator):
    pass


class OPTLargeOperator(OPTOperator):
    pass
