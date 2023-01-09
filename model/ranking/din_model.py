import torch
from torch import nn

from model.base_model import BaseConfig, BaseModel

from model.layer.mlp_layer import MLPLayer
from model.utils.activation import Dice


class DinConfig(BaseConfig):
    def __init__(
            self,
            hidden_size,
            attention_hidden_units,
            attention_hidden_activations,
            attention_output_activations,
            attention_dropout,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            batch_norm,
            use_softmax,
            **kwargs,
    ):
        super(DinConfig, self).__init__()

        self.hidden_size = hidden_size

        self.attention_hidden_units = attention_hidden_units
        self.attention_output_activations = attention_output_activations
        self.attention_hidden_activations = attention_hidden_activations
        self.attention_dropout = attention_dropout

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout

        self.batch_norm = batch_norm
        self.use_softmax = use_softmax


class DINAttention(nn.Module):
    def __init__(
            self,
            embedding_dim=64,
            attention_units=[32],
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rate=0,
            batch_norm=False,
            use_softmax=False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax

        if isinstance(hidden_activations, str) and hidden_activations.lower() == "dice":
            hidden_activations = [Dice(units) for units in attention_units]

        self.attention_layer = MLPLayer(
            input_dim=4 * embedding_dim,
            output_dim=1,
            hidden_units=attention_units,
            hidden_activations=hidden_activations,
            output_activation=output_activation,
            dropout_rates=dropout_rate,
            batch_norm=batch_norm,
            use_bias=True,
        )

    def forward(self, target_item, history_sequence, mask=None):
        # target_item: b x emd
        # history_sequence: b x len x emb
        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            target_item,
            history_sequence,
            target_item - history_sequence,
            target_item * history_sequence
        ], dim=-1)  # b x len x 4*emb
        attention_weight = self.attention_layer(attention_input.view(-1, 4 * self.embedding_dim))
        attention_weight = attention_weight.view(-1, seq_len)  # b x len
        if mask is not None:
            attention_weight = attention_weight * mask.float()
        if self.use_softmax:
            if mask is not None:
                attention_weight += -1.e9 * (1 - mask.float())
            attention_weight = attention_weight.softmax(dim=-1)
        output = (attention_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
        return output


class DinModel(BaseModel):
    def __init__(self, config: DinConfig):

        super().__init__()

        if isinstance(config.dnn_activations, str) and config.dnn_activations.lower() == "dice":
            config.dnn_activations = [Dice(units) for units in config.dnn_hidden_units]
        self.attention_layers = nn.ModuleList([
            DINAttention(
                config.hidden_size,
                attention_units=config.attention_hidden_units,
                hidden_activations=config.attention_hidden_activations,
                output_activation=config.attention_output_activations,
                dropout_rate=config.attention_dropout,
                batch_norm=config.batch_norm,
                use_softmax=config.use_softmax
            ) for _ in self.din_target_field])

        self.dnn = MLPLayer(
            input_dim=2 * config.hidden_size,
            output_dim=1,  # output hidden layer
            hidden_units=config.dnn_hidden_units,
            hidden_activations=config.dnn_activations,
            output_activation=None,
            dropout_rates=config.dnn_dropout,
            batch_norm=config.batch_norm,
            use_bias=True
        )

    def forward(self, item_embeds, behavior_embeds, behavior_mask):
        # behavior_embeds = behavior_embeds * behavior_mask.unsqueeze(dim=-1)  # [B, L, D]/
        # behavior_embeds, _ = self.attn(
        #     behavior_embeds,
        # )
        # concat_feature = torch.cat([item_embeds, behavior_embeds.squeeze()], dim=1)
        # output = self.dnn(concat_feature)
        # return output.squeeze()
        for idx, (target_field, sequence_field) in enumerate(zip(self.din_target_field,
                                                                 self.din_sequence_field)):
            target_emb = self.concat_embedding(target_field, feature_emb_dict)
            sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
            seq_field = np.array([sequence_field]).flatten()[0]  # pick a sequence field
            padding_idx = self.feature_map.feature_specs[seq_field]['padding_idx']
            mask = (X[:, self.feature_map.feature_specs[seq_field]["index"]].long() != padding_idx).float()
            pooling_emb = self.attention_layers[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(np.hstack([sequence_field]),
                                        pooling_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]
