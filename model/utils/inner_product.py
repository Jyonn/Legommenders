import torch
from torch import nn
from itertools import combinations


class InnerProductLayer(nn.Module):
    """ output: product_sum (bs x 1),
                bi_interaction (bs * dim),
                inner_product (bs x f^2/2),
                elementwise_product (bs x f^2/2 x emb_dim)
    """

    def __init__(self, num_fields=None, output="product_sum"):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in ["product_sum", "bi_interaction", "inner_product", "elementwise_product"]:
            raise ValueError("InnerProductLayer output={} is not supported.".format(output))
        if num_fields is None:
            if output in ["inner_product", "elementwise_product"]:
                raise ValueError("num_fields is required when InnerProductLayer output={}.".format(output))
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triange_mask = nn.Parameter(torch.triu(torch.ones(num_fields, num_fields), 1).byte(),
                                                   requires_grad=False)

    def forward(self, feature_emb):
        if self._output_type in ["product_sum", "bi_interaction"]:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)  # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "bi_interaction":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "elementwise_product":
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2
        elif self._output_type == "inner_product":
            inner_product_matrix = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
            flat_upper_triange = torch.masked_select(inner_product_matrix, self.upper_triange_mask)
            return flat_upper_triange.view(-1, self.interaction_units)
