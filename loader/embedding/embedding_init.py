from UniTok import UniDep, Vocab

from typing import Optional

from oba import Obj
from torch import nn

from loader.embedding.embedding_loader import EmbeddingLoader, EmbeddingInfo
from utils.printer import printer


class TransformEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, from_dim: int, to_dim: int):
        super(TransformEmbedding, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(from_dim, to_dim)

    def forward(self, indexes):
        return self.linear(self.embedding(indexes))


class EmbeddingInit:
    def __init__(
            self,
            order: list,
            depot: UniDep,
            hidden_size: int = 768,
            embedding_loader: EmbeddingLoader = None,
    ):
        self.print = printer.EMBEDDING__INIT_Cblue_
        self.order = order
        self.depot = depot
        self.hidden_size = hidden_size
        self.loader = embedding_loader

        self._table = None  # type: Optional[nn.ModuleDict]

    def register_vocab(self, vocab: Vocab):
        table = self.get_table()
        # table[vocab.name] = nn.Embedding(
        #     num_embeddings=vocab.get_size(),
        #     embedding_dim=self.hidden_size,
        # )
        self._register_vocab(
            table=self.get_table(),
            vocab_name=vocab.name,
            vocab_size=vocab.get_size(),
        )
        self.print(f'register vocab {vocab.name} ({vocab.get_size()}, {self.hidden_size})')

    def _register_vocab(self, table, vocab_name, vocab_size):
        embedding_info = self.loader.get_embedding(vocab_name)  # type: Optional[EmbeddingInfo]

        if embedding_info and embedding_info.embedding is not None:
            embedding = embedding_info.embedding
            is_frozen = "frozen" if embedding_info.frozen else "unfrozen"
            self.print(f'load {is_frozen} vocab: {vocab_name} {embedding.shape}')

            if int(embedding.shape[0]) != vocab_size:
                raise ValueError(f'not meet the expected vocab size {vocab_size}')

            table[vocab_name] = nn.Embedding.from_pretrained(embedding)
            table[vocab_name].weight.requires_grad = not embedding_info.frozen

            if int(embedding.shape[1]) != self.hidden_size:
                self.print(f'transform hidden size from {int(embedding.shape[1])} to {self.hidden_size}')
                table[vocab_name] = TransformEmbedding(
                    embedding=table[vocab_name],
                    from_dim=int(embedding.shape[1]),
                    to_dim=self.hidden_size
                )
            return

        self.print(f'create vocab {vocab_name} ({vocab_size}, {self.hidden_size})')
        table[vocab_name] = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size
        )

    def get_table(self) -> nn.ModuleDict:
        if self._table:
            return self._table

        vocabs = set()
        for col in self.order:
            vocabs.add(self.depot.get_vocab(col))

        table = dict()
        for vocab in vocabs:
            expected_vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)

            self._register_vocab(
                table=table,
                vocab_name=vocab,
                vocab_size=expected_vocab_size,
            )

        self._table = nn.ModuleDict(table)
        return self._table

    @classmethod
    def parse(cls, data, model, depot):
        embedding_loader = EmbeddingLoader()
        for embedding_info in data.token_embedding:
            embedding_loader.append(**Obj.raw(embedding_info))

        return cls(
            order=data.order,
            depot=depot,
            hidden_size=model.config.tok_embed_dim,
            embedding_loader=embedding_loader,
        )
