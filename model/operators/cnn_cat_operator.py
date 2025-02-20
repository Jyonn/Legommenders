import torch

from loader.env import Env
from model.inputer.simple_inputer import SimpleInputer
from model.operators.cnn_operator import CNNOperatorConfig, CNNOperator


class CNNCatOperatorConfig(CNNOperatorConfig):
    pass


class CNNCatOperator(CNNOperator):
    config_class = CNNCatOperatorConfig
    config: CNNCatOperatorConfig
    inputer_class = SimpleInputer

    num_columns: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_columns = len(self.lego_config.item_inputs)

    def forward(self, embeddings: dict, mask=None, **kwargs):
        output_list = []
        for col in embeddings:
            embedding = embeddings[col]
            if embedding.size()[1] > 1:
                embedding = self.cnn(embedding.permute(0, 2, 1))
                activation = self.activation(embedding.permute(0, 2, 1))
                masked_activation = activation * mask[col].unsqueeze(-1).to(Env.device)
                dropout = self.dropout(masked_activation)
                output = self.additive_attention(dropout, mask[col].to(Env.device))
            else:
                output = embedding.squeeze(1)
            output_list.append(output)

        outputs = torch.cat(output_list, dim=-1).to(Env.device)
        return outputs

    @property
    def output_dim(self):
        return self.config.hidden_size * self.num_columns

    def get_full_placeholder(self, sample_size):
        return torch.zeros(sample_size, self.output_dim, dtype=torch.float)
