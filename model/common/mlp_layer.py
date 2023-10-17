from torch import nn

from model.common.activation import get_activation


class MLPLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim=None,
            hidden_units=None,
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=0.0,
            batch_norm=False,
            use_bias=True
    ):
        super(MLPLayer, self).__init__()

        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units

        dense_layers = []
        for i in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[i + 1]))
            if hidden_activations[i]:
                dense_layers.append(hidden_activations[i])
            if dropout_rates[i] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[i]))

        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))

        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.dnn = nn.Sequential(*dense_layers)  # * used to unpack list

    def forward(self, inputs):
        return self.dnn(inputs)
