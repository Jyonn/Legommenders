import abc

from transformers.models.bert import BertModel

from embedder.base_embedder import BaseEmbedder


class BertEmbedder(BaseEmbedder, abc.ABC):
    def __init__(self, name):
        super().__init__(name)

        self.transformer: BertModel = BertModel.from_pretrained(name)

    def _get_embeddings(self):
        return self.transformer.embeddings.word_embeddings


class BertBaseEmbedder(BertEmbedder):
    def __init__(self):
        super().__init__('bert-base-uncased')


class BertLargeEmbedder(BertEmbedder):
    def __init__(self):
        super().__init__('bert-large-uncased')
