from loader.env import Env
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
    metadata = {
        "id": "Ada",
        "displayName": "Additive Attention User Encoder",
        "roles": ["user"],
        "explain": "Aggregates a click sequence into one user vector using additive attention.",
        "contract": {
            "inputShape": "tensor[B, L, input_dim]",
            "maskShape": "tensor[B, L]",
            "outputShape": "tensor[B, input_dim]",
            "outputKind": "vector",
            "outputDimPolicy": "config.input_dim",
        },
        "params": {
            "additive_hidden_size": {"min": 16, "step": 8, "description": "Hidden size used inside additive attention."},
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.input_dim,
            hidden_size=self.config.additive_hidden_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        mask = mask.to(Env.device)
        outputs = self.additive_attention(embeddings, mask)  # [B, D]
        return outputs

    @property
    def output_dim(self):
        return self.config.input_dim
