import torch
from torch import nn

from loader.global_setting import Setting
from model.common.mlp_layer import MLPLayer
from model.operator.pooling_operator import PoolingOperator
from model.recommenders.base_recommender import BaseRecommenderConfig, BaseRecommender


class DeepFMModelConfig(BaseRecommenderConfig):
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
    def __init__(self, input_dim):
        super(FactorizationMachine, self).__init__()
        # self.linear = nn.Linear(input_dim, 1, bias=False)
        # self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_embeddings):
        # lr_out = self.linear(input_embeddings) + self.bias
        sum_of_square = torch.sum(input_embeddings, dim=1) ** 2  # sum then square
        square_of_sum = torch.sum(input_embeddings ** 2, dim=1)  # square then sum
        bi_interaction = (sum_of_square - square_of_sum) * 0.5
        # print(lr_out.shape, bi_interaction.shape)
        return bi_interaction.sum(dim=-1, keepdim=True)


class DeepFMModel(BaseRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = PoolingOperator
    config_class = DeepFMModelConfig
    config: DeepFMModelConfig
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

        self.fm = FactorizationMachine(input_dim // 2)

        self.dnn = MLPLayer(
            input_dim=input_dim,
            output_dim=1,  # output hidden layer
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True
        )

    def predict(self, user_embedding, candidates, batch):
        labels = batch[self.label_col].to(Setting.device)

        squeezed_candidates = candidates.squeeze(1)  # [batch_size, hidden_size]
        input_embeddings = torch.stack([user_embedding, squeezed_candidates], dim=1)  # [batch_size, 2, hidden_size]

        fm_output = self.fm(input_embeddings)
        dnn_output = self.dnn(input_embeddings.flatten(start_dim=1))
        scores = (fm_output + dnn_output) / 2

        if Setting.status.is_testing:
            return scores

        scores = scores.squeeze(1)
        return nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
