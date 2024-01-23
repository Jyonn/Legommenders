import torch
from torch import nn

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictor
from model.predictors.dcn_predictor import DCNPredictorConfig


class GDCNPredictorConfig(DCNPredictorConfig):
    def __init__(
            self,
            sequential_mode: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.sequential_mode = sequential_mode


class GateCrossLayer(nn.Module):
    #  The core structure： gated cross layer.
    def __init__(self, input_dim, cross_layers=3):
        super().__init__()

        self.cross_layers = cross_layers

        self.w = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cross_layers)
        ])
        self.wg = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cross_layers)
        ])

        self.b = nn.ParameterList([nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cross_layers)])

        for i in range(cross_layers):
            nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cross_layers):
            xw = self.w[i](x) # Feature Crossing
            xg = self.activation(self.wg[i](x)) # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x


class GDCNPredictor(BasePredictor):
    config_class = GDCNPredictorConfig
    config: GDCNPredictorConfig

    def __init__(self, **kwargs):
        super(GDCNPredictor, self).__init__(**kwargs)

        output_dim = 1 if self.config.sequential_mode else None
        self.dnn = MLPLayer(
            input_dim=self.config.hidden_size * 2,
            output_dim=output_dim,
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True,
        )

        self.cross_net = GateCrossLayer(self.config.hidden_size * 2, self.config.cross_num)

        if not self.config.sequential_mode:
            output_dim = self.config.hidden_size * 2 + self.config.dnn_hidden_units[-1]
            self.prediction = nn.Linear(output_dim, 1)

    def predict(self, user_embeddings, item_embeddings):
        input_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)  # [batch_size, 2 * hidden_size]
        cross_output = self.cross_net(input_embeddings)

        if self.config.sequential_mode:
            dnn_output = self.dnn(cross_output)
            return dnn_output.flatten()

        dnn_output = self.dnn(input_embeddings)
        final_out = torch.cat([cross_output, dnn_output], dim=-1)
        return self.prediction(final_out).flatten()
