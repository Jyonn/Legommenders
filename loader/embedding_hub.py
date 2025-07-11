from typing import Dict, cast, Iterable

import numpy as np
import torch
from unitok import Vocab, Feature
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
        self.vocab_table = nn.ModuleDict()
        self.feature_table = nn.ModuleDict()

        self._pretrained_vocab_embeddings = dict()  # type: Dict[str, PretrainedEmbedding]
        self._pretrained_feature_embeddings = dict()  # type: Dict[str, PretrainedEmbedding]

    def load_pretrained_embedding(
            self,
            path,
            vocab_name=None,
            col_name=None,
            transformation=DEFAULT,
            transformation_dropout=None,
            frozen=True,
    ):
        if vocab_name is None and col_name is None:
            raise ValueError('vocab_name or col_name must be specified')
        if vocab_name is not None and col_name is not None:
            raise ValueError('only one of vocab_name and col_name can be specified')
        name = cast(str, vocab_name or col_name)

        embedding = np.load(path)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        embedding = nn.Embedding.from_pretrained(embedding)
        pnt(f'load pretrained embedding {name} of {embedding.weight.shape}:')

        if name == '<vocab_name>':
            raise ValueError(f'please specify the vocab name for the pretrained embedding in the embed config')

        if transformation not in self.pretrained_types:
            raise ValueError(f'invalid transformation type {transformation}, expected {self.pretrained_types}')
        if transformation == self.DEFAULT:
            pnt('--- use default transformation')
            transformation = self.transformation
        if transformation_dropout is None:
            pnt('--- use default transformation dropout')
            transformation_dropout = self.transformation_dropout
        pnt(f'--- pretrained transformation type: {transformation}')
        pnt(f'--- pretrained transformation dropout: {transformation_dropout}')

        if vocab_name is not None:
            target = self._pretrained_vocab_embeddings
            pnt(f'--- saved in pretrained vocab embeddings')
        else:
            target = self._pretrained_feature_embeddings
            pnt(f'--- saved in pretrained feature embeddings')

        target[name] = PretrainedEmbedding(
            embedder=embedding,
            transformation=transformation,
            transformation_dropout=transformation_dropout,
            frozen=frozen,
        )

    def _process_pretrained_embedding(self, name, size, pe: PretrainedEmbedding):
        frozen_str = "frozen" if pe.frozen else "unfrozen"
        pnt(f'--- load {frozen_str} vocab: {name} {pe.embedder.weight.shape}')

        if int(pe.embedder.weight.shape[0]) != size:
            raise ValueError(f'{name} not meet the expected vocab size {size}')

        pe.embedder.weight.requires_grad = not pe.frozen

        embedding_size = int(pe.embedder.weight.data.shape[1])

        if embedding_size != self.embedding_dim or self.transformation == self.LINEAR:
            pnt(f'--- transform {name} embedding from size {embedding_size} to {self.embedding_dim}')
            pe.embedder = Transformation(
                embedding=cast(nn.Embedding, pe.embedder),
                to_dimension=self.embedding_dim,
                transformation_dropout=pe.transformation_dropout,
            )

    def build_feature_embedding(self, feature: Feature):
        if feature.name in self.feature_table:
            return False

        if feature.name not in self._pretrained_feature_embeddings:
            return False

        pnt(f'build pretrained embedding for feature {feature.name} ({feature.tokenizer.vocab.size}, {self.embedding_dim})')

        pe = self._pretrained_feature_embeddings[feature.name]
        self._process_pretrained_embedding(feature.name, feature.tokenizer.vocab.size, pe)

        self.feature_table.add_module(feature.name, pe.embedder.to(Env.device))

        return True

    def build_vocab_embedding(self, vocab: Vocab):
        if vocab.name in self.vocab_table:
            return

        if vocab.name not in self._pretrained_vocab_embeddings:
            pnt(f'create vocab {vocab.name} ({vocab.size}, {self.embedding_dim})')
            self.vocab_table.add_module(vocab.name, nn.Embedding(
                num_embeddings=vocab.size,
                embedding_dim=self.embedding_dim
            ).to(Env.device))
            return

        pnt(f'build pretrained embedding for vocab {vocab.name} ({vocab.size}, {self.embedding_dim})')
        pe = self._pretrained_vocab_embeddings[vocab.name]
        self._process_pretrained_embedding(vocab.name, vocab.size, pe)

        self.vocab_table.add_module(vocab.name, pe.embedder.to(Env.device))

    def register_vocab(self, vocab: Vocab):
        if vocab.name in self._vocab_size:
            if self._vocab_size[vocab.name] != vocab.size:
                raise ValueError(f'conflict vocab {vocab.name}: {self._vocab_size[vocab.name]} vs {vocab.size}')
            return
        self._vocab_size[vocab.name] = vocab.size
        self.build_vocab_embedding(vocab)

    def register_ut(self, ut: LegoUT, used_cols: Iterable):
        for col in used_cols:
            feature = ut.meta.features[col]
            self.build_feature_embedding(feature)

            vocab = feature.tokenizer.vocab
            self.register_vocab(vocab)

    def __call__(self, vocab_name, col_name=None):
        if col_name in self.feature_table:
            return self.feature_table[col_name]
        return self.vocab_table[vocab_name]
