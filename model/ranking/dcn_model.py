import torch
from torch import nn

from model.base_model import BaseConfig, BaseModel


class DCNConfig(BaseConfig):
    def __init__(
            self,
            embed_dim,
    ):
        self.embed_dim = embed_dim


class DNN(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_units: list,
            dnn_dropout
    ):
        super(DNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_units = [self.embed_dim] + list(hidden_units)
        self.dropout = nn.Dropout(dnn_dropout)

        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i+1]) for i in range(len(self.hidden_units)-1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        for module in self.linear:
            fc = module(inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs


class CrossNet(nn.Module):
    def __init__(
            self,
            embed_dim,
            cross_layer_num=2,
            parameterization='vector',
    ):
        super(CrossNet, self).__init__()

        self.layer_num = cross_layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, embed_dim, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, embed_dim, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[0])

    def forward(self, inputs):
        inputs = inputs.unsqueeze(2)
        crossed = inputs
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                weights = torch.tensordot(crossed, self.kernels[i], dims=([1], [0]))
                dot = torch.matmul(inputs, weights)
                crossed = dot + self.bias[i] + crossed
            else:
                weights = torch.tensordot(self.kernels[i], crossed)
                dot = weights + self.bias[i]
                crossed = inputs * dot + crossed
        crossed = torch.squeeze(crossed, dim=2)
        return crossed


class DCN(BaseModel):
    def __init__(
            self,
            feat_size,
            embedding_size,
            linear_feature_columns,
            dnn_feature_columns,
            cross_num=2,
            cross_param='vector',
            dnn_hidden_units=(128, 128),
            init_std=0.0001,
            drop_rate=0.5
    ):
        super(DCN, self).__init__()
        self.feat_size = feat_size
        self.embedding_size = embedding_size
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        self.cross_param = cross_param
        self.drop_rate = drop_rate
        self.l2_reg = 0.00001

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

        self.dense_feature_columns = list(filter(lambda x:x[1]=='dense', dnn_feature_columns))
        self.sparse_feature_columns = list(filter(lambda x:x[1]=='sparse', dnn_feature_columns))

        self.feature_index = defaultdict(int)
        start = 0
        for feat in self.feat_size:
            self.feature_index[feat] = start
            start += 1


        inputs_dim = len(self.dense_feature_columns)+self.embedding_size*len(self.sparse_feature_columns)

        self.dnn = DNN(inputs_dim,self.dnn_hidden_units, 0.5)

        self.crossnet = CrossNet(inputs_dim, cross_layer_num=self.cross_num, parameterization=self.cross_param)
        self.dnn_linear = nn.Linear(inputs_dim+dnn_hidden_units[-1], 1, bias=False)

        dnn_hidden_units = [len(feat_size)] + list(dnn_hidden_units) + [1]
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i+1]) for i in range(len(dnn_hidden_units)-1)
        ])
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, X):

        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc

        dnn_input = torch.cat((dense_input, sparse_input), dim=1)

        # print('sparse input size', sparse_input.shape)
        # print('dense input size', dense_input.shape)
        # print('dnn input size', dnn_input.shape)

        deep_out = self.dnn(dnn_input)
        cross_out = self.crossnet(dnn_input)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)

        logit += self.dnn_linear(stack_out)
        #print('logit size', logit.shape)
        y_pred = torch.sigmoid(logit)
        #print('y_pred', y_pred.shape)
        return