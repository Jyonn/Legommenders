from typing import Dict, Union

from UniTok import Vocab
from torch import nn

from loader.embedding.embedding_init import TransformEmbedding
from loader.embedding.embedding_loader import EmbeddingInfo
from model_v2.utils.nr_depot import NRDepot
from utils.printer import printer, Color


class EmbeddingManager:
    def __init__(self, hidden_size):
        self._col_to_vocab = dict()
        self._vocab_to_size = dict()
        self._table = nn.ModuleDict()

        self.hidden_size = hidden_size
        self._pretrained = dict()  # type: Dict[str, EmbeddingInfo]

        self.print = printer[(self.__class__.__name__, '|', Color.YELLOW)]

    def get_table(self):
        return self._table

    def get(self, col, as_vocab=False):
        vocab = col if as_vocab else self._col_to_vocab[col]
        return self._table[vocab]

    def __call__(self, col, as_vocab=False):
        return self.get(col, as_vocab)

    def load_pretrained_embedding(self, vocab_name, **kwargs):
        self._pretrained[vocab_name] = EmbeddingInfo(**kwargs).load()

    def build_vocab_embedding(self, vocab_name, vocab_size):
        if vocab_name in self._table:
            return

        if vocab_name in self._pretrained:
            embedding_info = self._pretrained[vocab_name]
            embedding_weights = embedding_info.embedding

            is_frozen = "frozen" if embedding_info.frozen else "unfrozen"
            self.print(f'load {is_frozen} vocab: {vocab_name} {embedding_weights.shape}')

            if int(embedding_weights.shape[0]) != vocab_size:
                raise ValueError(f'not meet the expected vocab size {vocab_size}')

            embedding = nn.Embedding.from_pretrained(embedding_weights)
            embedding.weight.requires_grad = not embedding_info.frozen

            if int(embedding.shape[1]) != self.hidden_size:
                self.print(f'transform hidden size from {int(embedding.shape[1])} to {self.hidden_size}')
                embedding = TransformEmbedding(
                    embedding=embedding,
                    from_dim=int(embedding.shape[1]),
                    to_dim=self.hidden_size
                )
            self._table.add_module(vocab_name, embedding)
            return

        self.print(f'create vocab {vocab_name} ({vocab_size}, {self.hidden_size})')
        self._table.add_module(vocab_name, nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size
        ))

    def register_vocab(self, vocab_name: Union[str, Vocab], vocab_size=None):
        if isinstance(vocab_name, Vocab):
            vocab_name, vocab_size = vocab_name.name, vocab_name.get_size()
        else:
            assert vocab_size is not None, f'vocab size is required for {vocab_name}'

        self._col_to_vocab[vocab_name] = vocab_name
        self._vocab_to_size[vocab_name] = vocab_size
        self.build_vocab_embedding(vocab_name, vocab_size)

    def register_depot(self, nrd: NRDepot, skip_cols=None):
        depot, order = nrd.depot, nrd.order
        skip_cols = skip_cols or []

        for col in order:
            if col in skip_cols:
                self.print(f'skip col {col}')

            vocab_name = depot.get_vocab(col)
            vocab_size = depot.get_vocab_size(col)

            self._col_to_vocab[col] = vocab_name
            if vocab_name in self._vocab_to_size:
                assert self._vocab_to_size[vocab_name] == vocab_size, f'conflict vocab {vocab_name}'
                continue
            self._vocab_to_size[vocab_name] = vocab_size
            self.build_vocab_embedding(vocab_name, vocab_size)
