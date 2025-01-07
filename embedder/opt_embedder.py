import abc

from torch import nn
from transformers import OPTModel, AutoModel, AutoTokenizer

from embedder.base_embedder import BaseEmbedder
from utils.config_init import ModelInit


class OPTEmbedder(BaseEmbedder, abc.ABC):
    def __init__(self):
        super().__init__()
        self.transformer: OPTModel = AutoModel.from_pretrained(ModelInit.get(str(self)))
        self.tokenizer = AutoTokenizer.from_pretrained(ModelInit.get(str(self)))

    def _get_embeddings(self):
        vocab_size = self.tokenizer.vocab_size
        embeddings = self.transformer.decoder.embed_tokens.weight[:vocab_size]
        return nn.Embedding.from_pretrained(embeddings)


class OPTBaseEmbedder(OPTEmbedder):
    pass


class OPTLargeEmbedder(OPTEmbedder):
    pass
