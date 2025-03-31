from torch import nn

from model.operators.base_operator import BaseOperator


class LMOperator(BaseOperator):
    transformer: nn.Module

    def use_lm_cache(self):
        raise NotImplementedError

    @property
    def transformer_key(self):
        raise NotImplementedError

    @property
    def operator_name(self):
        return self.__class__.__name__.replace('Operator', '').lower()

    def get_layer_nums(self):
        raise NotImplementedError
