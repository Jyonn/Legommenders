import abc

from transformers.models.bert import BertModel

from embedder.base_embedder import BaseEmbedder
from utils.config_init import ModelInit


class BertEmbedder(BaseEmbedder, abc.ABC):
    def __init__(self):
        super().__init__()
        self.transformer: BertModel = BertModel.from_pretrained(ModelInit.get(str(self)))

    def _get_embeddings(self):
        return self.transformer.embeddings.word_embeddings


class BertBaseEmbedder(BertEmbedder):
    pass


class BertLargeEmbedder(BertEmbedder):
    pass
