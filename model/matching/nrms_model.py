import torch
from torch import nn
from torch.nn import functional as F

from model.base_model import BaseModel, BaseConfig
from model.layer.attention import AdditiveAttention


class NRMSConfig(BaseConfig):
    def __init__(
            self,
            tok_embed_dim,
            doc_embed_dim,
            ada_hidden_size,
            num_attention_heads,
            doc_embedding_dropout=0.2,
            doc_attention_dropout=0.1,
            seq_embedding_dropout=0.2,
            seq_attention_dropout=0.1,
            **kwargs,
    ):
        super(NRMSConfig, self).__init__(**kwargs)
        self.tok_embed_dim = tok_embed_dim
        self.doc_embed_dim = doc_embed_dim
        self.ada_hidden_size = ada_hidden_size
        self.num_attention_heads = num_attention_heads
        self.doc_embedding_dropout = doc_embedding_dropout
        self.doc_attention_dropout = doc_attention_dropout
        self.seq_embedding_dropout = seq_embedding_dropout
        self.seq_attention_dropout = seq_attention_dropout


class DocEncoder(nn.Module):
    def __init__(
            self,
            config: NRMSConfig,
    ):
        super(DocEncoder, self).__init__()
        self.config = config

        self.mha = nn.MultiheadAttention(
            embed_dim=self.config.tok_embed_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.doc_attention_dropout,
        )
        self.linear = nn.Linear(self.config.tok_embed_dim, self.config.doc_embed_dim)
        self.ada = AdditiveAttention(
            embed_dim=self.config.doc_embed_dim,
            hidden_size=self.config.ada_hidden_size
        )

    def forward(self, input_embeds, attention_mask):
        """

        @param input_embeds: [B, T, D]
        @param attention_mask: [B, T]
        @return: [B, T, encoder_size]
        """
        embedding = F.dropout(input_embeds, self.config.doc_embedding_dropout)
        embedding = embedding.permute(1, 0, 2)  # [T, B, D]

        attention_mask = (1 - attention_mask).bool()
        output, _ = self.mha(embedding, embedding, embedding, key_padding_mask=attention_mask)  # [T, B, D]
        # output, _ = self.mha(embedding, embedding, embedding)  # [T, B, D]
        output = F.dropout(output.permute(1, 0, 2))  # [B, T, D]
        output = self.linear(output)  # [B, T, encoder_size]
        output, _ = self.ada(output)
        return output


class NRMSNRLModel(BaseModel):
    config_class = NRMSConfig

    def __init__(self, config: NRMSConfig):
        super().__init__()
        self.config = config
        self.mha = nn.MultiheadAttention(
            embed_dim=config.doc_embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.seq_attention_dropout,
        )

        self.ada = AdditiveAttention(
            embed_dim=self.config.doc_embed_dim,
            hidden_size=self.config.ada_hidden_size
        )
        self.dropout = nn.Dropout(config.seq_embedding_dropout)

    def forward(
            self,
            clicks: torch.Tensor,
            candidates: torch.Tensor,
            click_mask: torch.Tensor,
    ) -> torch.Tensor:
        """forward
        Args:
            clicks (tensor): [num_user (B), num_click_docs (N), D]
            candidates (tensor): [num_user (B), num_candidate_docs (C), D]
            click_mask (tensor): [num_user (B), num_click_docs (c)]
        """
        click_mask = (1 - click_mask).bool()
        clicks = clicks.permute(1, 0, 2)  # [N, B, D]
        mha_doc_clicks, _ = self.mha(clicks, clicks, clicks, key_padding_mask=click_mask)
        # mha_doc_clicks, _ = self.mha(clicks, clicks, clicks)
        mha_doc_clicks = self.dropout(mha_doc_clicks.permute(1, 0, 2))  # [B, N, D]

        user_clicks, _ = self.ada(mha_doc_clicks)  # [B, D]
        matching_scores = torch.bmm(
            user_clicks.unsqueeze(1),  # [B, 1, D]
            candidates.permute(0, 2, 1)  # [B, D, C]
        ).squeeze(1)  # [B, C]

        return matching_scores


class NRMSModel(BaseModel):
    config_class = NRMSConfig

    def __init__(self, config: NRMSConfig):
        super().__init__()
        self.config = config
        self.doc_encoder = DocEncoder(config)
        self.click_encoder = NRMSNRLModel(config)

    def forward(
            self,
            doc_clicks: torch.Tensor,
            doc_candidates: torch.Tensor,
            doc_clicks_attention_mask: torch.Tensor,
            click_mask: torch.Tensor,
            doc_candidates_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """forward
        Args:
            doc_clicks (tensor): [num_user (B), num_click_docs (N), seq_len (T), D]
            doc_candidates (tensor): [num_user (B), num_candidate_docs (C), seq_len (T), D]
            doc_clicks_attention_mask (tensor): [num_user (B), num_click_docs (C), seq_len (T)]
            click_mask (tensor): [num_user (B), num_click_docs (c)]
            doc_candidates_attention_mask (tensor): [num_user (B), num_candidate_docs (C), seq_len (T)]
        """
        num_user, num_click, doc_len, embed_dim = doc_clicks.shape
        num_candidate = doc_candidates.shape[1]

        doc_clicks = doc_clicks.reshape(-1, doc_len, embed_dim)  # [-1, T, D]
        doc_candidates = doc_candidates.reshape(-1, doc_len, embed_dim)  # [-1, T, D]
        doc_clicks_attention_mask = doc_clicks_attention_mask.reshape(-1, doc_len)
        doc_candidates_attention_mask = doc_candidates_attention_mask.reshape(-1, doc_len)

        doc_clicks = self.doc_encoder(doc_clicks, doc_clicks_attention_mask)  # [-1, D]
        doc_candidates = self.doc_encoder(doc_candidates, doc_candidates_attention_mask)  # [-1, D]
        doc_clicks = doc_clicks.reshape(num_user, num_click, -1)  # [B, N, D]
        doc_candidates = doc_candidates.reshape(num_user, num_candidate, -1)  # [B, C, D]

        return self.click_encoder(doc_clicks, doc_candidates, click_mask)
