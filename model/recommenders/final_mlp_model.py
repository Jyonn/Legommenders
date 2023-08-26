import torch
from torch import nn

from loader.global_setting import Setting
from model.common.mlp_layer import MLPLayer
from model.operator.pooling_operator import PoolingOperator
from model.recommenders.base_recommender import BaseRecommender, BaseRecommenderConfig


class FinalMLPModelConfig(BaseRecommenderConfig):
    def __init__(
            self,
            mlp1_hidden_units=None,
            mlp1_hidden_activations="ReLU",
            mlp1_dropout=0,
            mlp1_batch_norm=False,
            mlp2_hidden_units=None,
            mlp2_hidden_activations="ReLU",
            mlp2_dropout=0,
            mlp2_batch_norm=False,
            use_fs=True,
            fs_hidden_units=[64],
            fs1_context=[],
            fs2_context=[],
            num_heads=1,
            embedding_regularizer=None,
            net_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if mlp2_hidden_units is None:
            mlp2_hidden_units = [64, 64, 64]
        if mlp1_hidden_units is None:
            mlp1_hidden_units = [64, 64, 64]
        self.mlp1_hidden_units = mlp1_hidden_units
        self.mlp1_hidden_activations = mlp1_hidden_activations
        self.mlp1_dropout = mlp1_dropout
        self.mlp1_batch_norm = mlp1_batch_norm
        self.mlp2_hidden_units = mlp2_hidden_units
        self.mlp2_hidden_activations = mlp2_hidden_activations
        self.mlp2_dropout = mlp2_dropout
        self.mlp2_batch_norm = mlp2_batch_norm
        self.use_fs = use_fs
        self.fs_hidden_units = fs_hidden_units
        self.fs1_context = fs1_context
        self.fs2_context = fs2_context
        self.num_heads = num_heads
        self.embedding_regularizer = embedding_regularizer
        self.net_regularizer = net_regularizer


# class FeatureSelection(nn.Module):
#     def __init__(
#             self,
#             feature_dim,
#             embedding_dim,
#             fs_hidden_units=[],
#             fs1_context=[],
#             fs2_context=[]
#     ):
#         super(FeatureSelection, self).__init__()
#         self.fs1_context = fs1_context
#         if len(fs1_context) == 0:
#             self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
#         else:
#             self.fs1_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
#                                                 required_feature_columns=fs1_context)
#         self.fs2_context = fs2_context
#         if len(fs2_context) == 0:
#             self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
#         else:
#             self.fs2_ctx_emb = FeatureEmbedding(feature_map, embedding_dim,
#                                                 required_feature_columns=fs2_context)
#         self.fs1_gate = MLPLayer(
#             input_dim=embedding_dim * max(1, len(fs1_context)),
#             output_dim=feature_dim,
#             hidden_units=fs_hidden_units,
#             hidden_activations="ReLU",
#             output_activation="Sigmoid",
#             batch_norm=False
#         )
#         self.fs2_gate = MLPLayer(
#             input_dim=embedding_dim * max(1, len(fs2_context)),
#             output_dim=feature_dim,
#             hidden_units=fs_hidden_units,
#             hidden_activations="ReLU",
#             output_activation="Sigmoid",
#             batch_norm=False
#         )
#
#     def forward(self, X, flat_emb):
#         if len(self.fs1_context) == 0:
#             fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
#         else:
#             fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
#         gt1 = self.fs1_gate(fs1_input) * 2
#         feature1 = flat_emb * gt1
#         if len(self.fs2_context) == 0:
#             fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
#         else:
#             fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
#         gt2 = self.fs2_gate(fs2_input) * 2
#         feature2 = flat_emb * gt2
#         return feature1, feature2


class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim,
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2),
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output


class FinalMLPModel(BaseRecommender):
    news_encoder_class = PoolingOperator
    user_encoder_class = PoolingOperator
    config_class = FinalMLPModelConfig
    config: FinalMLPModelConfig

    def __init__(self, input_dim=None, **kwargs):
        super(FinalMLPModel, self).__init__(**kwargs)
        if input_dim is None:
            input_dim = self.config.hidden_size
            if self.config.use_news_content and self.news_encoder.config.flatten:
                input_dim *= len(self.news_encoder.inputer.order)
        input_dim *= 2

        self.mlp1 = MLPLayer(
            input_dim=input_dim,
            output_dim=None,
            hidden_units=self.config.mlp1_hidden_units,
            hidden_activations=self.config.mlp1_hidden_activations,
            output_activation=None,
            dropout_rates=self.config.mlp1_dropout,
            batch_norm=self.config.mlp1_batch_norm,
        )
        self.mlp2 = MLPLayer(
            input_dim=input_dim,
            output_dim=None,
            hidden_units=self.config.mlp2_hidden_units,
            hidden_activations=self.config.mlp2_hidden_activations,
            output_activation=None,
            dropout_rates=self.config.mlp2_dropout,
            batch_norm=self.config.mlp2_batch_norm
        )
        self.use_fs = self.config.use_fs

        # if self.use_fs:
        #     self.fs_module = FeatureSelection(
        #         input_dim,
        #         self.config.hidden_size,
        #         self.config.fs_hidden_units,
        #         self.config.fs1_context,
        #         self.config.fs2_context
        #     )
        self.fusion_module = InteractionAggregation(
            self.config.mlp1_hidden_units[-1],
            self.config.mlp2_hidden_units[-1],
            output_dim=1,
            num_heads=self.config.num_heads,
        )

    def predict(self, user_embedding, candidates, batch):
        labels = batch[self.label_col].to(Setting.device)

        squeezed_candidates = candidates.squeeze(1)  # [batch_size, hidden_size]
        input_embeddings = torch.cat([user_embedding, squeezed_candidates], dim=1)  # [batch_size, 2 * hidden_size]

        # if self.use_fs:
        #     feat1, feat2 = self.fs_module(X, input_embeddings)
        # else:
        feat1, feat2 = input_embeddings, input_embeddings
        final_out = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))

        scores = self.prediction(final_out)  # [batch_size]

        if not Setting.status.is_training:
            return scores
        return nn.functional.binary_cross_entropy_with_logits(scores.squeeze(1), labels.float())

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        flat_emb = self.embedding_layer(X).flatten(start_dim=1)
        if self.use_fs:
            feat1, feat2 = self.fs_module(X, flat_emb)
        else:
            feat1, feat2 = flat_emb, flat_emb
        y_pred = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict