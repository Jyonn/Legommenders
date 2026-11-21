import abc

import torch
from transformers import GlmModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from model.common.glm_interface import ChatGLMModel
from model.operators.once_operator import OnceOperator


class GLMOperator(OnceOperator, abc.ABC):
    dtype = torch.bfloat16
    transformer: GlmModel | ChatGLMModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._uses_native_glm = isinstance(self.transformer, GlmModel)
        if not self._uses_native_glm:
            self.transformer = self.transformer.transformer
        self.transformer.set_input_embeddings(None)

        if self.transformer.config.hidden_size != self.config.input_dim:
            raise ValueError(f'In {self.classname}, hidden_size of transformer ({self.transformer.config.hidden_size}) '
                             f'does not match input_dim ({self.config.input_dim})')

        self._prepare_network()

    def _slice_transformer_layers(self):
        backbone = self.transformer if self._uses_native_glm else self.transformer.encoder
        backbone.layers = backbone.layers[self.config.tune_from + 1:]

    def get_layer_nums(self):
        if isinstance(self.transformer, GlmModel):
            return super().get_layer_nums()
        return int(self.transformer.config.num_layers)

    def _lora_encoder(self, peft_config):
        peft_config.target_modules = (
            {'q_proj', 'v_proj'} if self._uses_native_glm else {'query_key_value'}
        )
        super()._lora_encoder(peft_config)

    def _loop_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        if self._uses_native_glm:
            return self._forward(hidden_states, attention_mask, output_hidden_states=False)

        glm = self.transformer
        seq_length = hidden_states.size(1)
        rotary_pos_emb = glm.rotary_pos_emb(seq_length)
        rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        if attention_mask is not None and not attention_mask.all():
            attention_mask = glm.get_masks(attention_mask, None, padding_mask=attention_mask)

        for layer in glm.encoder.layers:
            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=None,
                use_cache=None
            )
            hidden_states, _ = layer_ret

        # Final layer norm.
        if glm.encoder.post_layer_norm:
            hidden_states = glm.encoder.final_layernorm(hidden_states)

        return hidden_states

    def _forward(
            self,
            inputs_embeds,
            attention_mask,
            output_hidden_states: bool = True,  # must be True, for caching hidden states
    ):
        if self._uses_native_glm:
            return super()._forward(inputs_embeds, attention_mask, output_hidden_states)
        output: BaseModelOutputWithPast = self.transformer(
            input_ids=attention_mask.long(),
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if output_hidden_states:
            return (*output.hidden_states[1:-1], output.last_hidden_state)
        return output.last_hidden_state


class GLM4TH9BOperator(GLMOperator):
    pass
