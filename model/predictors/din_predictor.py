import torch
from torch import nn

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class DINPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            dnn_hidden_units=None,
            dnn_activations="ReLU",
            attention_hidden_units=None,
            attention_output_activation=None,
            attention_dropout=0,
            net_dropout=0,
            batch_norm=False,
            din_use_softmax=False,
            embedding_regularizer=None,
            net_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if attention_hidden_units is None:
            attention_hidden_units = [self.hidden_size]
        if dnn_hidden_units is None:
            dnn_hidden_units = [self.hidden_size * 8, self.hidden_size * 2, self.hidden_size]
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.attention_hidden_units = attention_hidden_units
        self.attention_output_activation = attention_output_activation
        self.attention_dropout = attention_dropout
        self.net_dropout = net_dropout
        self.batch_norm = batch_norm
        self.din_use_softmax = din_use_softmax
        self.embedding_regularizer = embedding_regularizer
        self.net_regularizer = net_regularizer


class Dice(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + (1 - p) * self.alpha * X
        return output


class DINAttention(nn.Module):
    def __init__(self, config: DINPredictorConfig):
        super(DINAttention, self).__init__()
        self.config = config

        self.use_softmax = self.config.din_use_softmax
        hidden_activations = [Dice(units) for units in self.config.attention_hidden_units]
        self.attention_layer = MLPLayer(
            input_dim=4 * self.config.hidden_size,
            output_dim=1,
            hidden_units=self.config.attention_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=self.config.attention_output_activation,
            dropout_rates=self.config.attention_dropout,
            batch_norm=self.config.batch_norm,
            use_bias=True
        )

    def forward(self, candidate, clicks, mask=None):
        # candidate: b x emd
        # clicks: b x len x emb
        click_size = clicks.size(1)
        candidate = candidate.unsqueeze(1).expand(-1, click_size, -1)
        attention_input = torch.cat([
            candidate,
            clicks,
            candidate - clicks,
            candidate * clicks
        ], dim=-1)  # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.config.hidden_size))
        attention_weight = attention_weight.view(-1, click_size)  # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        if self.use_softmax:
            if mask is not None:
                attention_weight += -1.e9 * (1 - mask.float())
            attention_weight = attention_weight.softmax(dim=-1)
        output = (attention_weight.unsqueeze(-1) * clicks).sum(dim=1)
        return output


class DINPredictor(BasePredictor):
    allow_matching = False

    config_class = DINPredictorConfig
    config: DINPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.attention_layers = DINAttention(self.config)

        self.dnn = MLPLayer(
            input_dim=self.config.hidden_size,
            output_dim=1,
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.net_dropout,
            batch_norm=self.config.batch_norm,
            use_bias=True
        )

    def predict(self, user_embeddings, item_embeddings):
        user_embeddings, mask = user_embeddings['embedding'], user_embeddings['mask']
        pooling_embeddings = self.attention_layers(item_embeddings, user_embeddings, mask)
        scores = self.dnn(pooling_embeddings)
        return scores.flatten()
