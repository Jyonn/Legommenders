from torch import nn

from loader.global_setting import Setting
from model.common.attention import AdditiveAttention
from model.operator.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.concat_inputer import ConcatInputer


class AttentionOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            num_attention_heads: int = 8,
            attention_dropout: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout


class AttentionOperator(BaseOperator):

    config_class = AttentionOperatorConfig
    inputer_class = ConcatInputer
    config: AttentionOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # use multi-head attention to capture sequence-level information and
        # use additive attention to fuse the information

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.attention_dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Setting.device)
        outputs, _ = self.multi_head_attention(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            key_padding_mask=mask,
            need_weights=False,
        )  # [B, L, D]

        linear_outputs = self.linear(outputs)  # [B, L, D]
        return self.additive_attention(linear_outputs, mask)  # [B, D]
