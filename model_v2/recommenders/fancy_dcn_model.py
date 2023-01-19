import torch
from torch import nn

from loader.global_setting import Setting
from model.layer.mlp_layer import MLPLayer
from model_v2.operator.pooling_operator import PoolingOperator
from model_v2.recommenders.base_neg_recommender import BaseNegRecommender, BaseNegRecommenderConfig
from model_v2.recommenders.dcn_model import DCNModelConfig, CrossNet


class FancyDCNModelConfig(DCNModelConfig, BaseNegRecommenderConfig):
    # def __init__(
    #         self,
    #         **kwargs,
    # ):
    #     super().__init__(**kwargs)
    pass


class FancyDCNModel(BaseNegRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = PoolingOperator
    config_class = FancyDCNModelConfig
    config: FancyDCNModelConfig
    news_encoder: PoolingOperator
    user_encoder: PoolingOperator

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

    def predict(self, user_embedding, candidates, labels):
        shape = candidates.shape  # B, K+1, 2D
        flattened_candidates = candidates.view(-1, shape[-1])
        flattened_user_embedding = user_embedding.unsqueeze(1).expand(-1, shape[1], -1).contiguous().view(-1, user_embedding.shape[-1])
        # squeezed_candidates = candidates.squeeze(1)  # [batch_size, K+1, hidden_size]
        input_embeddings = torch.cat([flattened_candidates, flattened_user_embedding], dim=1)  # B*(K+1), 2D
        cross_output = self.cross_net(input_embeddings)
        dnn_output = self.dnn(input_embeddings)
        final_out = torch.cat([cross_output, dnn_output], dim=-1)
        scores = self.prediction(final_out)  # B*(K+1)

        if Setting.status.is_testing:
            return scores

        scores = scores.squeeze(1)
        labels = torch.zeros(shape[:-1], device=scores.device)
        labels[:, 0] = 1
        labels = labels.view(-1)
        return nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
