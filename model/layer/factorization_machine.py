from torch import nn
from .logistic_regression import LogisticRegression
from ..interactions import InnerProductLayer


class FactorizationMachine(nn.Module):
    def __init__(self, feature_map):
        super(FactorizationMachine, self).__init__()
        self.fm_layer = InnerProductLayer(feature_map.num_fields, output="product_sum")
        self.lr_layer = LogisticRegression(feature_map, use_bias=True)

    def forward(self, X, feature_emb):
        lr_out = self.lr_layer(X)
        fm_out = self.fm_layer(feature_emb)
        output = fm_out + lr_out
        return output
