from torch import nn

from loader.meta import Meta
from model.common.fastformer import FastformerModel, FastformerConfig
from model.operators.attention_operator import AttentionOperatorConfig
from model.operators.base_operator import BaseOperator
from model.inputer.concat_inputer import ConcatInputer


class FastformerOperatorConfig(AttentionOperatorConfig):
    def __init__(
            self,
            num_hidden_layers: int = 3,
            num_attention_heads: int = 8,
            hidden_dropout_prob: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob


class FastformerOperator(BaseOperator):
    config_class = FastformerOperatorConfig
    inputer_class = ConcatInputer
    config: FastformerOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # use multi-head attention to capture sequence-level information and
        # use additive attention to fuse the information

        self.fastformer = FastformerModel(FastformerConfig(
            hidden_size=self.config.input_dim,
            num_attention_heads=self.config.num_attention_heads,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            num_hidden_layers=self.config.num_hidden_layers,
            intermediate_size=self.config.input_dim * 4,
            max_position_embeddings=1024,
        ))

        self.linear = nn.Linear(self.config.input_dim, self.config.hidden_size)
        #
        # self.additive_attention = AdditiveAttention(
        #     embed_dim=self.config.hidden_size,
        #     hidden_size=self.config.hidden_size,
        # )

    def forward(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Meta.device)

        fastformer_output = self.fastformer(
            inputs_embeds=embeddings,
            attention_mask=mask,
        )
        # outputs = fastformer_output.last_hidden_state  # [B, L, D]
        #
        # outputs = self.linear(outputs)  # [B, L, D]
        # outputs = self.additive_attention(outputs, mask)  # [B, D]
        return self.linear(fastformer_output)  # [B, D]

        # return fastformer_output
