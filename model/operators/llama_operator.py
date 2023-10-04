import torch
from transformers import LlamaModel

from model.inputer.llm_concat_inputer import LlamaConcatInputer
from model.operators.base_llm_operator import BaseLLMOperator


class LlamaOperator(BaseLLMOperator):
    inputer_class = LlamaConcatInputer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = LlamaModel.from_pretrained(self.config.llm_dir)  # type: LlamaModel
        self.transformer.embed_tokens = None

        self.layer_split(self.transformer.config.num_hidden_layers)

    def _slice_transformer_layers(self):
        self.transformer.layers = self.transformer.layers[self.config.layer_split + 1:]

    def get_all_hidden_states(
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

        if llama.layers:
            hidden_states = llama.norm(hidden_states)

        all_hidden_states += (hidden_states,)

        return all_hidden_states
