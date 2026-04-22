from collections import OrderedDict

from model.inputer.concat_inputer import ConcatInputer
from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.simple_inputer import SimpleInputer


class NullOperatorConfig(BaseOperatorConfig):
    pass


class NullSimpleOperator(BaseOperator):
    inputer_class = SimpleInputer
    config_class = NullOperatorConfig
    config: NullOperatorConfig
    metadata = {
        "id": "NullSimple",
        "displayName": "Null Simple Operator",
        "roles": ["item", "user"],
        "explain": "Passes SimpleInputer embeddings and masks through as a structured dict.",
        "contract": {
            "inputShape": "dict[col -> tensor[N, Lc, input_dim]]",
            "maskShape": "dict[col -> tensor[N, Lc]]",
            "outputShape": "dict{embedding, mask}",
            "outputKind": "embedding-mask",
            "outputDimPolicy": "not a plain vector",
        },
    }

    def forward(self, embeddings: OrderedDict, mask: dict = None, **kwargs):
        return dict(
            embedding=embeddings,
            mask=mask
        )


class NullConcatOperator(NullSimpleOperator):
    inputer_class = ConcatInputer
    metadata = {
        "id": "NullConcat",
        "displayName": "Null Concat Operator",
        "roles": ["user"],
        "explain": "Passes concatenated embeddings and mask through, mainly for DIN-style target attention.",
        "contract": {
            "inputShape": "tensor[B, L, input_dim]",
            "maskShape": "tensor[B, L]",
            "outputShape": "dict{embedding: tensor[B, L, input_dim], mask: tensor[B, L]}",
            "outputKind": "embedding-mask",
            "outputDimPolicy": "not a plain vector",
            "compatiblePredictors": ["DINPredictor"],
        },
    }
