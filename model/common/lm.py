from torch import nn
from transformers import AutoModel


class BaseLM(nn.Module):
    def __init__(self, key):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(key, trust_remote_code=True)

    def get_layer_nums(self):
        return self.transformer.config.num_hidden_layers
