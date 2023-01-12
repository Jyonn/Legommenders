from torch import nn


class TransformEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, from_dim: int, to_dim: int):
        super(TransformEmbedding, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(from_dim, to_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, indexes):
        return self.dropout(self.linear(self.embedding(indexes)))
