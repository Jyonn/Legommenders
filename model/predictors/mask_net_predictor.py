import torch
from torch import nn

from model.common.activation import get_activation
from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class MaskNetPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            hidden_units,
            activations="ReLU",
            output_activation=None,
            dropout=0,
            layer_norm=True,
            embed_layer_norm=True,
            reduction_ratio=1,
            num_blocks=1,
            block_dim=64,
            sequential_mode=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.activations = activations
        self.output_activation = output_activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.reduction_ratio = reduction_ratio
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.sequential_mode = sequential_mode
        self.embed_layer_norm = embed_layer_norm


class MaskBlock(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            hidden_activation="ReLU",
            reduction_ratio=1,
            dropout_rate=0,
            layer_norm=True,
    ):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim)
        )
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, embeddings, hidden_states):
        mask = self.mask_layer(embeddings)
        return self.hidden_layer(mask * hidden_states)


class SerialMaskNet(nn.Module):
    def __init__(
            self, config: MaskNetPredictorConfig, input_dim):
        super(SerialMaskNet, self).__init__()
        if not isinstance(config.dropout, list):
            config.dropout = [config.dropout] * len(config.hidden_units)
        if not isinstance(config.activations, list):
            config.activations = [config.activations] * len(config.hidden_units)
        self.hidden_units = [input_dim] + config.hidden_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(
                MaskBlock(
                    input_dim,
                    self.hidden_units[idx],
                    self.hidden_units[idx + 1],
                    config.activations[idx],
                    config.reduction_ratio,
                    config.dropout[idx],
                    config.layer_norm,
                )
            )

        fc_layers = [nn.Linear(self.hidden_units[-1], 1)]
        if config.output_activation is not None:
            fc_layers.append(get_activation(config.output_activation))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, embeddings, hidden_states):
        output = hidden_states
        for idx in range(len(self.hidden_units) - 1):
            output = self.mask_blocks[idx](embeddings, output)
        if self.fc is not None:
            output = self.fc(output)
        return output


class ParallelMaskNet(nn.Module):
    def __init__(self, config: MaskNetPredictorConfig, input_dim):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = config.num_blocks
        self.mask_blocks = nn.ModuleList([
            MaskBlock(
                input_dim,
                input_dim,
                config.block_dim,
                config.activations,
                config.reduction_ratio,
                config.dropout,
                config.layer_norm
            ) for _ in range(self.num_blocks)])

        self.dnn = MLPLayer(
            input_dim=config.block_dim * self.num_blocks,
            output_dim=1,
            hidden_units=config.hidden_units,
            hidden_activations=config.activations,
            output_activation=config.output_activation,
            dropout_rates=config.dropout
        )

    def forward(self, embeddings, hidden_states):
        block_out = []
        for i in range(self.num_blocks):
            block_out.append(self.mask_blocks[i](embeddings, hidden_states))
        concat_out = torch.cat(block_out, dim=-1)
        return self.dnn(concat_out)


class MaskNetPredictor(BasePredictor):
    config_class = MaskNetPredictorConfig
    config: MaskNetPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.config.sequential_mode:
            self.mask_net = SerialMaskNet(
                config=self.config,
                input_dim=self.config.hidden_size * 2,
            )
        else:
            self.mask_net = ParallelMaskNet(
                config=self.config,
                input_dim=self.config.hidden_size * 2
            )
        if self.config.embed_layer_norm:
            self.emb_norm = nn.ModuleList(nn.LayerNorm(self.config.hidden_size) for _ in range(2))
        else:
            self.emb_norm = None

    def predict(self, user_embeddings, item_embeddings):
        input_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)  # [batch_size, 2 * hidden_size]
        if self.emb_norm is not None:
            feat_list = input_embeddings.chunk(self.num_fields, dim=1)
            hidden_states = torch.cat([self.emb_norm[i](feat) for i, feat in enumerate(feat_list)], dim=1)
        else:
            hidden_states = input_embeddings
        return self.mask_net.forward(
            input_embeddings.flatten(start_dim=1),
            hidden_states.flatten(start_dim=1)
        ).flatten()
