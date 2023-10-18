from torch import nn

from loader.meta import Meta
from model.common.attention import AdditiveAttention
from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.concat_inputer import ConcatInputer


class AttentionOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            num_attention_heads: int = 8,
            attention_dropout: float = 0.1,
            additive_hidden_size: int = 256,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.additive_hidden_size = additive_hidden_size


class AttentionOperator(BaseOperator):
    config_class = AttentionOperatorConfig
    inputer_class = ConcatInputer
    config: AttentionOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.config.input_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(self.config.input_dim, self.config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.additive_hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Meta.device)

        outputs, _ = self.multi_head_attention(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            key_padding_mask=(1 - mask).bool(),
            need_weights=False,
        )  # [B, L, D]
        linear_outputs = self.linear(outputs)  # [B, L, D]
        output = self.additive_attention(linear_outputs, mask)  # [B, D]

        return output
