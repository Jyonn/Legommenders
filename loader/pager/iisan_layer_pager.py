import torch

from model.inputer.concat_inputer import ConcatInputer
from loader.pager.base_pager import BasePager


class IISANLayerPager(BasePager):
    def __init__(
            self,
            inputer: ConcatInputer,
            num_layers: int,
            hidden_size: int,
            **kwargs
    ):
        super().__init__(desc='Language Model Layer Caching for IISAN', **kwargs)

        self.inputer = inputer
        self.num_layers = num_layers

        self.final_states = torch.zeros(
            len(self.contents),
            num_layers,
            hidden_size,
        )

    def get_features(self, content, index) -> dict:
        return dict(
            inputs_embeds=self.inputer.get_embeddings(content).cpu().detach(),
            attention_mask=self.inputer.get_mask(content).cpu().detach(),
        )

    def combine(self, slices, features, output):
        self.final_states[slices] = output.cpu().detach()
