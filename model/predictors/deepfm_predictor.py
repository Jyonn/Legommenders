import torch
from torch import nn

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class DeepFMPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            dnn_batch_norm,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm


class FactorizationMachine(nn.Module):
    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, input_embeddings):
        sum_of_square = torch.sum(input_embeddings, dim=1) ** 2
        square_of_sum = torch.sum(input_embeddings ** 2, dim=1)
        bi_interaction = (sum_of_square - square_of_sum) * 0.5
        return bi_interaction.sum(dim=-1, keepdim=True)


class DeepFMPredictor(BasePredictor):
    config_class = DeepFMPredictorConfig
    config: DeepFMPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fm = FactorizationMachine()

        self.dnn = MLPLayer(
            input_dim=self.config.hidden_size * 2,
            output_dim=1,  # output hidden layer
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True
        )

    def predict(self, user_embeddings, item_embeddings):
        input_embeddings = torch.stack([user_embeddings, item_embeddings], dim=1)  # [B, 2, D]
        fm_output = self.fm(input_embeddings)
        dnn_output = self.dnn(input_embeddings.flatten(start_dim=1))
        scores = (fm_output + dnn_output) / 2
        return scores.flatten()
