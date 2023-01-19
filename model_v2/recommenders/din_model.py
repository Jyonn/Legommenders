import torch
from torch import nn

from loader.global_setting import Setting
from model.layer.mlp_layer import MLPLayer
from model_v2.operator.null_operator import NullConcatOperator
from model_v2.operator.pooling_operator import PoolingOperator
from model_v2.recommenders.base_recommender import BaseRecommenderConfig, BaseRecommender


class DINModelConfig(BaseRecommenderConfig):
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
            attention_hidden_units = [64]
        if dnn_hidden_units is None:
            dnn_hidden_units = [512, 128, 64]
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
    def __init__(self, input_dim, config: DINModelConfig):
        super(DINAttention, self).__init__()
        self.config = config

        self.hidden_size = input_dim
        self.use_softmax = self.config.din_use_softmax
        hidden_activations = [Dice(units) for units in self.config.attention_hidden_units]
        self.attention_layer = MLPLayer(
            input_dim=4 * self.hidden_size,
            output_dim=1,
            hidden_units=self.config.attention_hidden_units,
            hidden_activations=hidden_activations,
            output_activation=self.config.attention_output_activation,
            dropout_rates=self.config.attention_dropout,
            batch_norm=self.config.batch_norm,
            use_bias=True
        )

    def forward(self, candidate, clicks, mask=None):
        # target_item: b x emd
        # history_sequence: b x len x emb
        click_size = clicks.size(1)
        candidate = candidate.expand(-1, click_size, -1)
        attention_input = torch.cat([
            candidate,
            clicks,
            candidate - clicks,
            candidate * clicks
        ], dim=-1)  # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.hidden_size))
        attention_weight = attention_weight.view(-1, click_size)  # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        if self.use_softmax:
            if mask is not None:
                attention_weight += -1.e9 * (1 - mask.float())
            attention_weight = attention_weight.softmax(dim=-1)
        output = (attention_weight.unsqueeze(-1) * clicks).sum(dim=1)
        return output


class DINModel(BaseRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = NullConcatOperator
    config_class = DINModelConfig
    config: DINModelConfig
    news_encoder: PoolingOperator
    user_encoder: NullConcatOperator
    use_neg_sampling = False

    def __init__(self, input_dim=None, **kwargs):
        super().__init__(**kwargs)

        if input_dim is None:
            input_dim = self.config.hidden_size
            if self.config.use_news_content and self.news_encoder.config.flatten:
                input_dim *= len(self.news_encoder.inputer.order)

        self.attention_layers = DINAttention(input_dim, self.config)

        self.dnn = MLPLayer(
            input_dim=input_dim,
            output_dim=1,
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.net_dropout,
            batch_norm=self.config.batch_norm,
            use_bias=True
        )

    def predict(self, user_embedding, candidates, labels):
        user_embedding, mask = user_embedding['embedding'], user_embedding['mask']
        # print(user_embedding.shape, candidates.shape, mask.shape)

        pooling_embedding = self.attention_layers(candidates, user_embedding, mask)
        scores = self.dnn(pooling_embedding)

        if Setting.status.is_testing:
            return scores

        scores = scores.squeeze(1)
        return nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
