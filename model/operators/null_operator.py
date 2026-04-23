from collections import OrderedDict

from model.inputer.concat_inputer import ConcatInputer
from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.simple_inputer import SimpleInputer


class NullOperatorConfig(BaseOperatorConfig):
    pass


class NullSimpleOperator(BaseOperator):
    """
    Legommenders:
    {
      "summary": "Passes SimpleInputer embeddings and masks through as a structured dict.",
      "roles": ["item", "user"],
      "contract": {
        "outputs": [
          {"name": "payload", "kind": "embedding-mask", "shape": {"type": "dict", "keys": ["embedding", "mask"]}, "dimPolicy": "not a plain vector"}
        ]
      }
    }
    """
    inputer_class = SimpleInputer
    config_class = NullOperatorConfig
    config: NullOperatorConfig

    def forward(self, embeddings: OrderedDict, mask: dict = None, **kwargs):
        return dict(
            embedding=embeddings,
            mask=mask
        )


class NullConcatOperator(NullSimpleOperator):
    """
    Legommenders:
    {
      "summary": "Passes concatenated embeddings and mask through, mainly for DIN-style target attention.",
      "roles": ["user"],
      "contract": {
        "outputs": [
          {"name": "payload", "kind": "embedding-mask", "shape": {"type": "dict", "fields": {
            "embedding": {"type": "tensor", "axes": ["B", "L", "input_dim"]},
            "mask": {"type": "tensor", "axes": ["B", "L"]}
          }}, "dimPolicy": "not a plain vector"}
        ],
        "compatiblePredictors": ["DINPredictor"]
      }
    }
    """
    inputer_class = ConcatInputer
