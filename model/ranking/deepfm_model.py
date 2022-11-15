from model.base_model import BaseModel, BaseConfig
from model.layer.mlp_layer import MLPLayer


class DeepFMConfig(BaseConfig):
    def __init__(
            self,
            embed_dim,
            **kwargs,
    ):
        super(DeepFMConfig, self).__init__(**kwargs)
        self.embed_dim = embed_dim


class DeepFM(BaseModel):
    def __init__(self, config: DeepFMConfig, **kwargs):
        super().__init__(**kwargs)
        input_dim = config.embed_dim * len(config.column)

        self.fm = FactorizationMachine(feature_map)
        self.mlp = MLPLayer(
            input_dim=input_dim,
            output_dim=1,
            hidden_units=config.dnn_hidden_units,
            hidden_activations=config.dnn_activations,
            output_activation=None,
            dropout_rates=config.dnn_dropout,
            batch_norm=config.dnn_batch_norm,
        )

    def forward(self, input_embeddings):
        """
        Inputs: [X,y]
        """
        X, y, g = inputs
        feature_emb = self.embedding_layer(X)
        y_pred = self.fm(X, feature_emb)
        y_pred += self.mlp(feature_emb.flatten(start_dim=1))
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred, "group": g}
        return return_dict
