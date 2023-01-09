from model_v2.common.base_config import BaseConfig


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
