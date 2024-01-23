import torch
from torch import nn

from model.inputer.concat_inputer import ConcatInputer
from model.operators.base_operator import BaseOperator, BaseOperatorConfig


class PolyAttentionOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            num_context_codes: int = 32,
            context_code_dim: int = 200,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_context_codes = num_context_codes
        self.context_code_dim = context_code_dim


class PolyAttentionOperator(BaseOperator):
    config_class = PolyAttentionOperatorConfig
    inputer_class = ConcatInputer
    config: PolyAttentionOperatorConfig
    allow_caching = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.linear = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.context_code_dim,
            bias=False
        )
        self.context_codes = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    self.config.num_context_codes,
                    self.config.context_code_dim
                ),
                gain=nn.init.calculate_gain('tanh')
            )
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embeddings, mask=None, **kwargs):
        """
        @param embeddings: batch_size, seq_len, hidden_size
        @param mask: batch_size, seq_len
        @return: batch_size, num_context_codes, hidden_size
        """
        proj = torch.tanh(self.linear(embeddings))
        weights = torch.matmul(proj, self.context_codes.T)
        weights = weights.permute(0, 2, 1)
        weights = weights.masked_fill_(~mask.unsqueeze(dim=1).bool(), 1e-30)
        weights = self.softmax(weights)
        poly_repr = torch.matmul(weights, embeddings)

        return poly_repr
