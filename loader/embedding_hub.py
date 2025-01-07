from typing import Dict, cast, Iterable, Optional

import numpy as np
import torch
from unitok import Vocab
from pigmento import pnt
from torch import nn

from loader.env import Env
from loader.ut.lego_ut import LegoUT


class Transformation(nn.Module):
    def __init__(
            self,
            embedding: nn.Embedding,
            to_dimension: int,
            transformation_dropout,
    ):
        super(Transformation, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(embedding.weight.data.shape[1], to_dimension)
        self.dropout = nn.Dropout(transformation_dropout)

    def forward(self, indexes):
        return self.dropout(self.linear(self.embedding(indexes)))


class PretrainedEmbedding:
    def __init__(
            self,
            embedder,
            transformation,
            transformation_dropout,
            frozen,
    ):
        self.embedder: nn.Module = embedder
        self.transformation: str = transformation
        self.transformation_dropout: float = transformation_dropout
        self.frozen: bool = frozen


class EmbeddingHub:
    LINEAR = 'linear'
    AUTO = 'auto'
    DEFAULT = 'default'

    global_types = {LINEAR, AUTO}  # types for the embedding hub
    pretrained_types = {DEFAULT, LINEAR, AUTO}  # types for specific pretrained embedding, default means follow the global type

    def __init__(
            self,
            embedding_dim,
            transformation,
            transformation_dropout: float,
    ):
        if transformation not in self.global_types:
            raise ValueError(f'invalid transformation type {transformation}, expected {self.global_types}')

        pnt(f'global transformation type: {transformation}')
        pnt(f'global transformation dropout: {transformation_dropout}')

        self.embedding_dim = embedding_dim
        self.transformation = transformation
        self.transformation_dropout = transformation_dropout

        self._vocab_size = dict()
        self.table = nn.ModuleDict()

        self._pretrained_embeddings = dict()  # type: Dict[str, PretrainedEmbedding]

    def load_pretrained_embedding(
            self,
            vocab_name,
            path,
            transformation=DEFAULT,
            transformation_dropout=None,
            frozen=True,
    ):
        embedding = np.load(path)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = nn.Embedding.from_pretrained(embedding)
        pnt(f'load pretrained embedding {vocab_name} of {embedding.weight.shape}')

        if vocab_name == '<vocab_name>':
            raise ValueError(f'please specify the vocab name for the pretrained embedding in the embed config')

        if transformation not in self.pretrained_types:
            raise ValueError(f'invalid transformation type {transformation}, expected {self.pretrained_types}')
        if transformation == self.DEFAULT:
            pnt('use default transformation')
            transformation = self.transformation
        if transformation_dropout is None:
            pnt('use default transformation dropout')
            transformation_dropout = self.transformation_dropout
        pnt(f'pretrained transformation type: {transformation}')
        pnt(f'pretrained transformation dropout: {transformation_dropout}')

        self._pretrained_embeddings[vocab_name] = PretrainedEmbedding(
            embedder=embedding,
            transformation=transformation,
            transformation_dropout=transformation_dropout,
            frozen=frozen,
        )

    def build_vocab_embedding(self, vocab: Vocab):
        if vocab.name in self.table:
            return

        if vocab.name not in self._pretrained_embeddings:
            pnt(f'create vocab {vocab.name} ({vocab.size}, {self.embedding_dim})')
            self.table.add_module(vocab.name, nn.Embedding(
                num_embeddings=vocab.size,
                embedding_dim=self.embedding_dim
            ).to(Env.device))
            return

        pe = self._pretrained_embeddings[vocab.name]

        frozen_str = "frozen" if pe.frozen else "unfrozen"
        pnt(f'load {frozen_str} vocab: {vocab.name} {pe.embedder.weight.shape}')

        if int(pe.embedder.weight.shape[0]) != vocab.size:
            raise ValueError(f'{vocab.name} not meet the expected vocab size {vocab.size}')

        pe.embedder.weight.requires_grad = not pe.frozen

        embedding_size = int(pe.embedder.weight.data.shape[1])

        if embedding_size != self.embedding_dim or self.transformation == self.LINEAR:
            pnt(f'transform {vocab.name} embedding from size {embedding_size} to {self.embedding_dim}')
            pe.embedder = Transformation(
                embedding=cast(nn.Embedding, pe.embedder),
                to_dimension=self.embedding_dim,
                transformation_dropout=pe.transformation_dropout,
            )

        self.table.add_module(vocab.name, pe.embedder.to(Env.device))

    def register_vocab(self, vocab: Vocab):
        if vocab.name in self._vocab_size:
            if self._vocab_size[vocab.name] != vocab.size:
                raise ValueError(f'conflict vocab {vocab.name}: {self._vocab_size[vocab.name]} vs {vocab.size}')
            return
        self._vocab_size[vocab.name] = vocab.size
        self.build_vocab_embedding(vocab)

    def register_ut(self, ut: LegoUT, used_cols: Iterable):
        for col in used_cols:
            vocab = ut.meta.jobs[col].tokenizer.vocab
            self.register_vocab(vocab)

    def __call__(self, vocab_name):
        return self.table[vocab_name]

    # def register_depot(self, nrd: DataHub, skip_cols=None):
    #     depot, order = nrd.ut, nrd.input_cols
    #     skip_cols = skip_cols or []
    #     skip_vocabs = [depot.meta.jobs[col].tokenizer.vocab.name for col in skip_cols]
    #
    #     for col in order:
    #         vocab = depot.meta.jobs[col].tokenizer.vocab
    #
    #         if vocab.name in skip_vocabs:
    #             pnt(f'skip col {col}')
    #             continue
    #
    #         pnt(f'build mapping {col} -> {vocab.name}')
    #         if vocab.name in self._vocab_size:
    #             assert self._vocab_size[vocab.name] == vocab.size, f'conflict vocab {vocab.key}'
    #             continue
    #         self._vocab_size[vocab.name] = vocab.size
    #         self.build_vocab_embedding(vocab)