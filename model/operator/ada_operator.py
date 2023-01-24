from loader.global_setting import Setting
from model.common.attention import AdditiveAttention
from model.operator.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.concat_inputer import ConcatInputer


class AdaOperatorConfig(BaseOperatorConfig):
    pass


class AdaOperator(BaseOperator):

    config_class = AdaOperatorConfig
    inputer_class = ConcatInputer
    config: AdaOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Setting.device)
        outputs = self.additive_attention(embeddings, mask)  # [B, D]
        return outputs
