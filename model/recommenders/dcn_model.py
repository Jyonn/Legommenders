import torch
from torch import nn

from loader.global_setting import Setting
from model.common.mlp_layer import MLPLayer
from model.operator.pooling_operator import PoolingOperator
from model.recommenders.base_recommender import BaseRecommenderConfig, BaseRecommender


class DCNModelConfig(BaseRecommenderConfig):
    def __init__(
            self,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            dnn_batch_norm,
            cross_num,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm
        self.cross_num = cross_num


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


class DCNModel(BaseRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = PoolingOperator
    config_class = DCNModelConfig
    config: DCNModelConfig
    news_encoder: PoolingOperator
    user_encoder: PoolingOperator
    use_neg_sampling = False

    def __init__(self, input_dim=None, **kwargs):
        super().__init__(**kwargs)

        if input_dim is None:
            input_dim = self.config.hidden_size
            if self.config.use_news_content and self.news_encoder.config.flatten:
                input_dim *= len(self.news_encoder.inputer.order)
        input_dim *= 2

        self.dnn = MLPLayer(
            input_dim=input_dim,
            output_dim=None,  # output hidden layer
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True
        )

        self.cross_net = CrossNet(input_dim, self.config.cross_num)

        output_dim = input_dim + self.config.dnn_hidden_units[-1]
        self.prediction = nn.Linear(output_dim, 1)

    def predict(self, user_embedding, candidates, batch):
        labels = batch[self.label_col].to(Setting.device)

        squeezed_candidates = candidates.squeeze(1)  # [batch_size, hidden_size]
        input_embeddings = torch.cat([user_embedding, squeezed_candidates], dim=1)  # [batch_size, 2 * hidden_size]
        cross_output = self.cross_net(input_embeddings)
        dnn_output = self.dnn(input_embeddings)
        final_out = torch.cat([cross_output, dnn_output], dim=-1)
        scores = self.prediction(final_out)  # [batch_size]

        if not Setting.status.is_training:
            return scores
        return nn.functional.binary_cross_entropy_with_logits(scores.squeeze(1), labels.float())
