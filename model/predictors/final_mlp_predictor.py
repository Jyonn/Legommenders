import torch
from torch import nn

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class FinalMLPPredictorConfig(BasePredictorConfig):
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
            fs_hidden_units=None,
            fs1_context=None,
            fs2_context=None,
            num_heads=1,
            embedding_regularizer=None,
            net_regularizer=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if mlp2_hidden_units is None:
            mlp2_hidden_units = [self.hidden_size] * 3
        if mlp1_hidden_units is None:
            mlp1_hidden_units = [self.hidden_size] * 3
        if fs_hidden_units is None:
            fs_hidden_units = [self.hidden_size]
        if fs1_context is None:
            fs1_context = []
        if fs2_context is None:
            fs2_context = []

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


class FinalMLPPredictor(BasePredictor):
    config_class = FinalMLPPredictorConfig
    config: FinalMLPPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mlp1 = MLPLayer(
            input_dim=self.config.hidden_size * 2,
            output_dim=None,
            hidden_units=self.config.mlp1_hidden_units,
            hidden_activations=self.config.mlp1_hidden_activations,
            output_activation=None,
            dropout_rates=self.config.mlp1_dropout,
            batch_norm=self.config.mlp1_batch_norm,
        )
        self.mlp2 = MLPLayer(
            input_dim=self.config.hidden_size * 2,
            output_dim=None,
            hidden_units=self.config.mlp2_hidden_units,
            hidden_activations=self.config.mlp2_hidden_activations,
            output_activation=None,
            dropout_rates=self.config.mlp2_dropout,
            batch_norm=self.config.mlp2_batch_norm
        )
        self.use_fs = self.config.use_fs

        self.fusion_module = InteractionAggregation(
            self.config.mlp1_hidden_units[-1],
            self.config.mlp2_hidden_units[-1],
            output_dim=1,
            num_heads=self.config.num_heads,
        )

    def predict(self, user_embeddings, item_embeddings):
        input_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)  # [batch_size, 2 * hidden_size]
        feat1, feat2 = input_embeddings, input_embeddings
        scores = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        return scores.flatten()
