import torch
from torch import nn

from model.base_model import BaseConfig, BaseModel
from model.layer.mlp_layer import MLPLayer


class DCNConfig(BaseConfig):
    def __init__(
            self,
            embed_dim,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            dnn_batch_norm,
            cross_num,
            columns,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm

        self.cross_num = cross_num
        self.columns = columns


class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, input_embeddings, hidden_states):
        return self.weight(hidden_states) * input_embeddings + self.bias


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(
            CrossInteractionLayer(input_dim) for _ in range(self.num_layers)
        )

    def forward(self, input_embeddings):
        output_embeddings = input_embeddings
        for cross_net in self.cross_net:
            output_embeddings = output_embeddings + cross_net(input_embeddings, output_embeddings)
        return output_embeddings


class DCNModel(BaseModel):
    config_class = DCNConfig

    def __init__(
            self,
            config: DCNConfig,
    ):
        super(DCNModel, self).__init__()
        input_dim = config.embed_dim * len(config.columns)

        self.dnn = MLPLayer(
            input_dim=input_dim,
            output_dim=None,  # output hidden layer
            hidden_units=config.dnn_hidden_units,
            hidden_activations=config.dnn_activations,
            output_activation=None,
            dropout_rates=config.dnn_dropout,
            batch_norm=config.dnn_batch_norm,
            use_bias=True
        )

        self.cross_net = CrossNet(input_dim, config.cross_num)

        output_dim = input_dim + config.dnn_hidden_units[-1]
        self.prediction = nn.Linear(output_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input_embeddings):
        cross_output = self.cross_net(input_embeddings)
        dnn_output = self.dnn(input_embeddings)
        final_out = torch.cat([cross_output, dnn_output], dim=-1)
        return self.activation(self.prediction(final_out))
