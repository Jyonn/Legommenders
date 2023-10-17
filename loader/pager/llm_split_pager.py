import os

import numpy as np
import torch

from model.inputer.natural_concat_inputer import NaturalConcatInputer
from loader.pager.base_pager import BasePager


class LLMSplitPager(BasePager):
    def __init__(
            self,
            inputer: NaturalConcatInputer,
            layers: list,
            hidden_size: int,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.inputer = inputer
        self.layers = layers

        self.final_features = torch.zeros(
            len(layers),
            len(self.contents),
            self.inputer.max_sequence_len,
            hidden_size,
        )
        self.final_masks = torch.zeros(
            len(self.contents),
            inputer.max_sequence_len
        )

    def get_features(self, content, index) -> dict:
        return dict(
            hidden_states=self.inputer.get_embeddings(content).cpu().detach(),
            attention_mask=self.inputer.get_mask(content).cpu().detach(),
        )

    def combine(self, slices, features, output):
        for index, layer in enumerate(self.layers):
            self.final_features[index][slices] = output[layer].cpu().detach()
        self.final_masks[slices] = features['attention_mask'].cpu().detach()

    def store(self, store_dir):
        os.makedirs(store_dir, exist_ok=True)
        for index, layer in enumerate(self.layers):
            np.save(os.path.join(store_dir, f'layer_{layer}.npy'), self.final_features[index].numpy())
        np.save(os.path.join(store_dir, f'mask.npy'), self.final_masks.numpy())
