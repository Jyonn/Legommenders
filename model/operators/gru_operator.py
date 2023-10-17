import torch
from torch import nn

from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.concat_inputer import ConcatInputer


class GRUOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            columns: list,
            num_layers: int = 1,
            use_news_content: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_columns = len(columns)
        self.num_layers = num_layers
        self.use_news_content = use_news_content


class GRUOperator(BaseOperator):

    config_class = GRUOperatorConfig
    inputer_class = ConcatInputer
    config: GRUOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        input_size = self.config.hidden_size
        if self.config.use_news_content:
            input_size *= self.config.num_columns

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.linear = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=input_size,
        )

    def forward(self, embeddings, mask=None, **kwargs):

        embeddings = embeddings.to(torch.float32)
        lengths = mask.cpu().numpy().sum(axis=1)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, last_hidden = self.gru(packed_sequence)
        # last_hidden = last_hidden.squeeze(0)
        last_hidden = last_hidden[-1]

        return self.linear(last_hidden)
