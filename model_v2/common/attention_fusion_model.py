from torch import nn

from model.layer.attention import AdditiveAttention
from model_v2.common.base_config import BaseConfig
from model_v2.common.base_model import BaseOperator
from model_v2.inputer.cat_inputer import CatInputer
from utils.structure import Structure


class AttentionFusionConfig(BaseConfig):
    def __init__(
            self,
            num_attention_heads: int = 8,
            attention_dropout: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout


class AttentionFusionOperator(BaseOperator):
    config_class = AttentionFusionConfig
    inputer = CatInputer

    def __init__(self, config: AttentionFusionConfig):
        super().__init__(config=config)
        # use multi-head attention to capture sequence-level information and
        # use additive attention to fuse the information

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=config.hidden_size,
            hidden_size=config.hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        # first, use multi-head attention to capture sequence-level information

        # input_embeds: [B, L, D]
        # attention_mask: [B, L]
        # output: [B, L, D]
        outputs, _ = self.multi_head_attention(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            key_padding_mask=mask,
            need_weights=False,
        )  # [B, L, D]

        outputs = self.linear(outputs)  # [B, L, D]
        outputs = self.additive_attention(outputs, mask)  # [B, D]

        return outputs
