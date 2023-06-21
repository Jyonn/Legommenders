from typing import Dict, Union

import torch
from UniTok import Vocab
from torch import nn

from loader.embedding.embedding_loader import EmbeddingInfo
from model.utils.nr_depot import NRDepot
from utils.printer import printer, Color


class TransformEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, from_dim: int, to_dim: int):
        super(TransformEmbedding, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(from_dim, to_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, indexes):
        return self.dropout(self.linear(self.embedding(indexes)))


class TransformMultiEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, to_dim: int, hidden_dim: int = None):
        # embedding: [V, L, D] -> [V, L * D]
        super(TransformMultiEmbedding, self).__init__()
        embedding = embedding.view(embedding.shape[0], -1)
        self.embedding = nn.Embedding.from_pretrained(embedding)
        if hidden_dim:
            self.linear = nn.Sequential(
                nn.Linear(embedding.shape[1], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, to_dim),
            )
        else:
            self.linear = nn.Linear(embedding.shape[1], to_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, indexes):
        return self.dropout(self.linear(self.embedding(indexes)))


class EmbeddingManager:
    def __init__(self, hidden_size, same_dim_transform):
        self._col_to_vocab = dict()
        self._vocab_to_size = dict()
        self._table = nn.ModuleDict()

        self.hidden_size = hidden_size
        self.same_dim_transform = same_dim_transform
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
        self.print(f'load pretrained embedding {vocab_name} of {self._pretrained[vocab_name].embedding.shape}')

    def build_vocab_embedding(self, vocab_name, vocab_size):
        if vocab_name in self._table:
            return

        if vocab_name in self._pretrained:
            embedding_info = self._pretrained[vocab_name]
            embedding_weights = embedding_info.embedding

            is_frozen = "frozen" if embedding_info.frozen else "unfrozen"
            self.print(f'load {is_frozen} vocab: {vocab_name} {embedding_weights.shape}')

            if int(embedding_weights.shape[0]) != vocab_size:
                raise ValueError(f'{vocab_name} not meet the expected vocab size {vocab_size}')

            if embedding_weights.dim() == 3:
                embedding = TransformMultiEmbedding(embedding_weights, self.hidden_size)
                embedding.embedding.weight.requires_grad = not embedding_info.frozen
                self.print(f'load multi-embedding {embedding_weights.shape}')
            else:
                embedding = nn.Embedding.from_pretrained(embedding_weights)
                embedding.weight.requires_grad = not embedding_info.frozen

                embedding_size = int(embedding.weight.data.shape[1])
                if embedding_size != self.hidden_size or self.same_dim_transform:
                    self.print(f'transform hidden size from {embedding_size} to {self.hidden_size}')
                    embedding = TransformEmbedding(
                        embedding=embedding,
                        from_dim=embedding_size,
                        to_dim=self.hidden_size
                    )
                else:
                    self.print(f'keep transform size {embedding_size}')
            self._table.add_module(vocab_name, embedding)
            return

        self.print(f'create vocab {vocab_name} ({vocab_size}, {self.hidden_size})')
        self._table.add_module(vocab_name, nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size
        ))

    def clone_vocab(self, col_name, clone_col_name):
        self._col_to_vocab[col_name] = self._col_to_vocab[clone_col_name]

    def register_vocab(self, vocab_name: Union[str, Vocab], vocab_size=None):
        if isinstance(vocab_name, Vocab):
            vocab_name, vocab_size = vocab_name.name, len(vocab_name)
        else:
            assert vocab_size is not None, f'vocab size is required for {vocab_name}'

        self._col_to_vocab[vocab_name] = vocab_name
        self._vocab_to_size[vocab_name] = vocab_size
        self.build_vocab_embedding(vocab_name, vocab_size)

    def register_depot(self, nrd: NRDepot, skip_cols=None):
        depot, order = nrd.depot, nrd.order
        skip_cols = skip_cols or []
        skip_vocabs = [depot.get_vocab(col) for col in skip_cols]

        for col in order:
            vocab_name = depot.get_vocab(col)
            vocab_size = depot.get_vocab_size(col)

            if vocab_name in skip_vocabs:
                self.print(f'skip col {col}')
                continue

            self._col_to_vocab[col] = vocab_name
            self.print(f'build mapping {col} -> {vocab_name}')
            if vocab_name in self._vocab_to_size:
                assert self._vocab_to_size[vocab_name] == vocab_size, f'conflict vocab {vocab_name}'
                continue
            self._vocab_to_size[vocab_name] = vocab_size
            self.build_vocab_embedding(vocab_name, vocab_size)
