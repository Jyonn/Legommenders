from torch import nn

from loader.env import Env
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
    metadata = {
        "id": "Attention",
        "displayName": "Multihead Attention Encoder",
        "roles": ["item", "user"],
        "explain": "Runs self-attention over a concatenated feature sequence, then pools with additive attention.",
        "contract": {
            "inputShape": "tensor[N, L, input_dim]",
            "maskShape": "tensor[N, L]",
            "outputShape": "tensor[N, hidden_size]",
            "outputKind": "vector",
            "outputDimPolicy": "config.hidden_size",
        },
        "params": {
            "num_attention_heads": {"min": 1, "step": 1, "description": "Number of self-attention heads."},
            "attention_dropout": {"min": 0, "max": 0.9, "step": 0.05, "description": "Dropout inside multi-head attention."},
            "additive_hidden_size": {"min": 16, "step": 8, "description": "Hidden size used inside additive attention pooling."},
        },
    }

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
        mask = mask.to(Env.device)

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
