import torch

from model.inputer.single_column_inputer import SingleColumnInputer
from model.operators.base_operator import BaseOperator, BaseOperatorConfig


class SCSimpleOperatorConfig(BaseOperatorConfig):
    pass


class SCSimpleOperator(BaseOperator):
    config_class = SCSimpleOperatorConfig
    config: SCSimpleOperatorConfig
    inputer_class = SingleColumnInputer
    inputer: SingleColumnInputer

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

    def forward(self, embeddings, mask=None, **kwargs):
        return embeddings

    def get_full_placeholder(self, sample_size):
        max_length = self.inputer.ut.meta.features[self.inputer.inputs[0]].max_len
        return torch.zeros(sample_size, max_length, self.config.hidden_size, dtype=torch.float)

    @property
    def output_dim(self):
        return self.config.input_dim


class SCFlattenOperator(SCSimpleOperator):
    flatten_mode = True
