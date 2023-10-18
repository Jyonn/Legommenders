from loader.meta import Meta
from model.common.attention import AdditiveAttention
from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.concat_inputer import ConcatInputer


class AdaOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            additive_hidden_size: int = 256,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.additive_hidden_size = additive_hidden_size


class AdaOperator(BaseOperator):
    config_class = AdaOperatorConfig
    inputer_class = ConcatInputer
    config: AdaOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.additive_hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Meta.device)
        outputs = self.additive_attention(embeddings, mask)  # [B, D]
        return outputs
