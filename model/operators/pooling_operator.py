from collections import OrderedDict

import torch

from loader.meta import Meta
from model.operators.base_operator import BaseOperator, BaseOperatorConfig
from model.inputer.simple_inputer import SimpleInputer


class PoolingOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            flatten: bool = False,
            max_pooling: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.flatten = flatten
        self.max_pooling = max_pooling


class PoolingOperator(BaseOperator):
    inputer_class = SimpleInputer
    config_class = PoolingOperatorConfig
    config: PoolingOperatorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, embeddings: OrderedDict, mask: dict = None, **kwargs):
        assert mask is not None, 'mask is required for pooling fusion'

        if isinstance(embeddings, torch.Tensor):
            assert isinstance(mask, torch.Tensor)
            embeddings = dict(temp=embeddings)
            mask = dict(temp=mask)
        elif isinstance(mask, torch.Tensor):
            assert len(embeddings) == 1
            key = list(embeddings.keys())[0]
            mask = {key: mask}

        pooled_embeddings = dict()

        for col in embeddings:
            col_mask = mask[col].to(Meta.device)
            col_embedding = embeddings[col]
            col_embedding = col_embedding * col_mask.unsqueeze(-1)
            if self.config.max_pooling:
                pooled_embeddings[col], _ = col_embedding.max(dim=1)
            else:
                pooled_embeddings[col] = col_embedding.sum(dim=1) / (col_mask.sum(dim=1).unsqueeze(-1) + 1e-8)

        if self.config.flatten:
            order = embeddings.keys()
            return torch.cat([pooled_embeddings[col] for col in order], dim=-1)  # B, D * K

        stack = torch.stack([pooled_embeddings[col] for col in embeddings.keys()], dim=1)  # B, K, D
        if self.config.max_pooling:
            return stack.max(dim=1)[0]  # B, D
        return stack.mean(dim=1)  # B, D
