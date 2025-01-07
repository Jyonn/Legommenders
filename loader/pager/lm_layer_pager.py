import torch

from model.inputer.concat_inputer import ConcatInputer
from loader.pager.base_pager import BasePager


class LMLayerPager(BasePager):
    def __init__(
            self,
            inputer: ConcatInputer,
            layers: list,
            hidden_size: int,
            **kwargs
    ):
        super().__init__(desc='Language Model Layer Caching', **kwargs)

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
            inputs_embeds=self.inputer.get_embeddings(content).cpu().detach(),
            attention_mask=self.inputer.get_mask(content).cpu().detach(),
        )

    def combine(self, slices, features, output):
        for index, layer in enumerate(self.layers):
            self.final_features[index][slices] = output[layer].cpu().detach()
        self.final_masks[slices] = features['attention_mask'].cpu().detach()
