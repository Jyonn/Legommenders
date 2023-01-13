from torch import nn

from model_v2.operator.base_operator import BaseOperator, BaseOperatorConfig
from model_v2.inputer.concat_inputer import ConcatInputer


class GRUOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            columns: list,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_columns = len(columns)


class GRUOperator(BaseOperator):

    config_class = GRUOperatorConfig
    inputer_class = ConcatInputer
    config: GRUOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = nn.GRU(
            input_size=self.config.hidden_size * self.config.num_columns,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.linear = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size * self.config.num_columns,
        )

    def forward(self, embeddings, mask=None, **kwargs):
        lengths = mask.sum(dim=1).cpu()
        embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, last_hidden = self.gru(embeddings)
        last_hidden = last_hidden.squeeze(0)
        return self.linear(last_hidden)
