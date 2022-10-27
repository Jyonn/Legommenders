import torch
from torch import nn
from torch.nn import functional as F

from model.base_model import BaseModel, BaseConfig
from model.utils.attention import AdditiveAttention


class NRMSConfig(BaseConfig):
    def __init__(
            self,
            embedding_dim,
            doc_encoder_size,
            additive_attention_hidden_size,
            num_attention_heads,
            embedding_dropout,
            doc_attention_dropout,
            seq_attention_dropout,
            user_dropout=0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.doc_encoder_size = doc_encoder_size
        self.additive_attention_hidden_size = additive_attention_hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout = embedding_dropout
        self.doc_attention_dropout = doc_attention_dropout
        self.seq_attention_dropout = seq_attention_dropout
        self.user_dropout = user_dropout


class DocEncoder(nn.Module):
    config = NRMSConfig

    def __init__(
            self,
            config: NRMSConfig,
    ):
        super(DocEncoder, self).__init__()
        self.config = config

        self.mha = nn.MultiheadAttention(
            embed_dim=self.config.embedding_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.doc_attention_dropout,
        )
        self.linear = nn.Linear(self.config.embedding_dim, self.config.doc_encoder_size)
        self.ada = AdditiveAttention(
            embedding_dim=self.config.doc_encoder_size,
            hidden_size=self.config.additive_attention_hidden_size
        )

    def forward(self, input_embeds):
        """

        @param input_embeds: [B, T, D]
        @return: [B, T, encoder_size]
        """
        embedding = F.dropout(input_embeds, self.config.embedding_dropout)
        embedding = embedding.permute(1, 0, 2)

        output, _ = self.mha(embedding, embedding, embedding)
        output = F.dropout(output.permute(1, 0, 2))
        output = self.linear(output)
        return self.ada(output)


class NRMSModel(BaseModel):
    config_class = NRMSConfig

    def __init__(self, config: NRMSConfig):
        super().__init__()
        self.config = config
        self.doc_encoder = DocEncoder(config)
        self.mha = nn.MultiheadAttention(
            embed_dim=config.doc_encoder_size,
            num_heads=config.num_attention_heads,
            dropout=config.seq_attention_dropout,
        )

        # self.linear = nn.Linear(self.config.doc_encoder_size, self.config.doc_encoder_size)
        self.ada = AdditiveAttention(self.config.doc_encoder_size, self.config.additive_attention_hidden_size)
        self.dropout = nn.Dropout(config.user_dropout)

    def forward(
            self,
            doc_clicks: torch.Tensor,
            doc_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """forward
        Args:
            doc_clicks (tensor): [num_user (B), num_click_docs (N), seq_len (T), D]
            doc_candidates (tensor): [num_user (B), num_candidate_docs (C), seq_len (T), D]
        """
        num_user, num_click, doc_len, embedding_dim = doc_clicks.shape
        num_candidate = doc_candidates.shape[1]
        doc_clicks = doc_clicks.reshape(-1, doc_len, embedding_dim)  # [-1, T, D]
        doc_candidates = doc_candidates.reshape(-1, doc_len, embedding_dim)  # [-1, T, D]

        doc_clicks = self.doc_encoder(doc_clicks)  # [-1, D]
        doc_candidates = self.doc_encoder(doc_candidates)  # [-1, D]
        doc_clicks = doc_clicks.reshape(num_user, num_click, -1)  # [B, N, D]
        doc_candidates = doc_candidates.reshape(num_user, num_candidate, -1)  # [B, C, D]

        doc_clicks = doc_clicks.permute(1, 0, 2)  # [N, B, D]
        mha_doc_clicks, _ = self.mha(doc_clicks, doc_clicks, doc_clicks)
        mha_doc_clicks = self.dropout(mha_doc_clicks.permute(1, 0, 2))  # [B, N, D]

        # click_repr = self.linear(click_output)
        user_clicks, _ = self.ada(mha_doc_clicks)  # [B, D]
        matching_scores = torch.bmm(
            user_clicks.unsqueeze(1),  # [B, 1, D]
            doc_candidates.permute(0, 2, 1)  # [B, D, C]
        ).squeeze(1)  # [B, C]

        return torch.sigmoid(matching_scores)
