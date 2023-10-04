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

    def forward(self, embeddings: OrderedDict, mask: dict = None, **kwargs):
        return dict(
            embedding=embeddings,
            mask=mask
        )


class NullConcatOperator(NullSimpleOperator):
    inputer_class = ConcatInputer
