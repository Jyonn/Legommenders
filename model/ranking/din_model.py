import torch

from model.base_model import BaseConfig, BaseModel
from model.layer.attention import AdditiveAttention

from model.layer.mlp_layer import MLPLayer


class DinConfig(BaseConfig):
    def __init__(
            self,
            hidden_size,
            attention_hidden_size,
            attention_bias,
            attention_batch_norm,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            dnn_batch_norm,
            **kwargs,
    ):
        super(DinConfig, self).__init__()

        self.hidden_size = hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.attention_bias = attention_bias
        self.attention_batch_norm = attention_batch_norm

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm


class DinModel(BaseModel):
    def __init__(self, config: DinConfig):

        super().__init__()

        self.attn = AdditiveAttention(
            embed_dim=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
        )
        self.dnn = MLPLayer(
            input_dim=2 * config.hidden_size,
            output_dim=1,  # output hidden layer
            hidden_units=config.dnn_hidden_units,
            hidden_activations=config.dnn_activations,
            output_activation=None,
            dropout_rates=config.dnn_dropout,
            batch_norm=config.dnn_batch_norm,
            use_bias=True
        )

    def forward(self, item_embeds, behavior_embeds, behavior_mask):
        behavior_embeds = behavior_embeds * behavior_mask.unsqueeze(dim=-1)  # [B, L, D]/
        behavior_embeds, _ = self.attn(
            behavior_embeds,
        )
        concat_feature = torch.cat([item_embeds, behavior_embeds.squeeze()], dim=1)
        output = self.dnn(concat_feature)
        return output.squeeze()
