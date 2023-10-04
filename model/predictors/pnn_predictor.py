import torch
from torch import nn

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class PNNPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            dnn_batch_norm,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm


class InnerProductInteraction(nn.Module):
    def __init__(self, num_fields):
        super(InnerProductInteraction, self).__init__()
        self.interaction_units = int(num_fields * (num_fields - 1) / 2)
        self.triu_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).bool(), requires_grad=False)

    def forward(self, feature_emb):
        inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
        triu_values = torch.masked_select(inner_product_matrix, self.triu_mask)
        return triu_values.view(-1, self.interaction_units)


class PNNPredictor(BasePredictor):
    config_class = PNNPredictorConfig
    config: PNNPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inner_product_layer = InnerProductInteraction(num_fields=2)

        self.dnn = MLPLayer(
            input_dim=self.config.hidden_size + 1,
            output_dim=1,  # output hidden layer
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True
        )

    def predict(self, user_embeddings, item_embeddings):
        input_embeddings = torch.stack([user_embeddings, item_embeddings], dim=1)
        inner_products = self.inner_product_layer(input_embeddings)
        input_embeddings = torch.cat([input_embeddings.flatten(start_dim=1), inner_products], dim=1)
        scores = self.dnn(input_embeddings)
        return scores.flatten()
