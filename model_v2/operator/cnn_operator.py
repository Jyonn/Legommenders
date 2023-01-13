import torch
from torch import nn

from loader.global_setting import Setting
from model.layer.attention import AdditiveAttention
from model_v2.inputer.simple_inputer import SimpleInputer
from model_v2.operator.base_operator import BaseOperatorConfig, BaseOperator


class CNNOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            kernel_size: int = 3,
            dropout: float = 0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.dropout = dropout


class CNNOperator(BaseOperator):
    config_class = CNNOperatorConfig
    config: CNNOperatorConfig
    inputer_class = SimpleInputer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cnn = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.kernel_size,
            padding='same',
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

        self.additive_attention = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
        )

    def forward(self, embeddings: dict, mask=None, **kwargs):
        output_list = []
        output_mask = []
        for col in embeddings:
            embedding = embeddings[col]
            if embedding.size()[1] > 1:
                embedding = self.cnn(embedding.permute(0, 2, 1))
                embedding = self.activation(embedding.permute(0, 2, 1))
                embedding *= mask[col].unsqueeze(-1).to(Setting.device)
                embedding = self.dropout(embedding)
            output_list.append(embedding)
            output_mask.append(mask[col])

        outputs = torch.cat(output_list, dim=1).to(Setting.device)
        mask = torch.cat(output_mask, dim=1).to(Setting.device)
        outputs = self.additive_attention(outputs, mask)
        return outputs
