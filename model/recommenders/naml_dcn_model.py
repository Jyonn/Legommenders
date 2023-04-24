import torch
from torch import nn

from loader.global_setting import Setting
from model.common.mlp_layer import MLPLayer
from model.operator.ada_operator import AdaOperator
from model.operator.cnn_operator import CNNOperator
from model.recommenders.base_neg_recommender import BaseNegRecommender
from model.recommenders.dcn_model import CrossNet, DCNModelConfig


class NAMLDCNModel(BaseNegRecommender):
    news_encoder_class = CNNOperator
    user_encoder_class = AdaOperator
    config_class = DCNModelConfig
    config: DCNModelConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        if Setting.status.is_testing:
            return scores

        scores = scores.squeeze(1)
        return nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
