from torch import nn

from model.layer.attention import AdditiveAttention
from model_v2.common.attention_fusion_config import AttentionFusionConfig
from model_v2.common.base_model import BaseBatch, BaseModel
from model_v2.inputer.concat_inputer import ConcatInputer


class AttentionBatch(BaseBatch):
    def __init__(self, batch: dict):
        super().__init__(batch)

        self.input_embeds = batch['input_embeds']
        self.attention_mask = batch['attention_mask']  # [batch_size, seq_len], 1 for valid, 0 for padding


class AttentionFusionModel(BaseModel):
    batcher = AttentionBatch
    inputer = ConcatInputer

    def __init__(self, config: AttentionFusionConfig):
        super().__init__(config=config)
        # use multi-head attention to capture sequence-level information and
        # use additive attention to fuse the information

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.additive_attention = AdditiveAttention(
            embed_dim=config.hidden_size,
            hidden_size=config.hidden_size,
        )

    def forward(self, batch: AttentionBatch):
        # first, use multi-head attention to capture sequence-level information

        # input_embeds: [B, L, D]
        # attention_mask: [B, L]
        # output: [B, L, D]
        outputs = self.multi_head_attention(
            query=batch.input_embeds,
            key=batch.input_embeds,
            value=batch.input_embeds,
            key_padding_mask=batch.attention_mask,
            need_weights=False
        )  # [B, L, D]

        outputs = self.linear(outputs)  # [B, L, D]
        outputs = self.additive_attention(outputs, batch.attention_mask)  # [B, D]

        return outputs
