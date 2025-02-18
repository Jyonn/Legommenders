# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for Legommenders Project:
# The following code has been modified and extended as part of the Legommenders project.
# Changes were made by Qijiong Liu (2025) to adapt the original implementation for
# new functionality related to content-based recommender systems.
#
# =========================================================================

import torch
from torch import nn

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class DCNv2PredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            model_structure="parallel",
            use_low_rank_mixture=False,
            low_rank=32,
            num_experts=4,
            stacked_dnn_hidden_units=None,
            parallel_dnn_hidden_units=None,
            dnn_activations="ReLU",
            cross_num=3,
            dnn_dropout=0,
            dnn_batch_norm=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        if stacked_dnn_hidden_units is None:
            stacked_dnn_hidden_units = [self.hidden_size] * 3
        if parallel_dnn_hidden_units is None:
            parallel_dnn_hidden_units = [self.hidden_size] * 3
        if model_structure not in ["crossnet_only", "stacked", "parallel", "stacked_parallel"]:
            raise ValueError(f"model_structure must be one of ['crossnet_only', 'stacked', 'parallel', 'stacked_parallel'], got {model_structure}")
        self.model_structure = model_structure
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.stacked_dnn_hidden_units = stacked_dnn_hidden_units
        self.parallel_dnn_hidden_units = parallel_dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.cross_num = cross_num
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim)
                                          for _ in range(self.num_layers))

    def forward(self, input_embeddings):
        final_embeddings = input_embeddings  # b x dim
        for i in range(self.num_layers):
            final_embeddings = final_embeddings + input_embeddings * self.cross_layers[i](final_embeddings)
        return final_embeddings


class CrossNetMix(nn.Module):
    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # V: (in_features, low_rank)
        self.V_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # C: (low_rank, low_rank)
        self.C_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for i in range(self.layer_num)])
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(in_features, 1))) for i in range(self.layer_num)])
        # self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class DCNv2Predictor(BasePredictor):
    config_class = DCNv2PredictorConfig
    config: DCNv2PredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.config.use_low_rank_mixture:
            self.cross_net = CrossNetMix(
                in_features=self.config.hidden_size * 2,
                layer_num=self.config.cross_num,
                low_rank=self.config.low_rank,
                num_experts=self.config.num_experts
            )
        else:
            self.cross_net = CrossNetV2(
                input_dim=self.config.hidden_size * 2,
                num_layers=self.config.cross_num
            )

        output_dim = None
        if 'stack' in self.config.model_structure:
            self.stacked_dnn = MLPLayer(
                input_dim=self.config.hidden_size * 2,
                output_dim=None,  # output hidden layer
                hidden_units=self.config.stacked_dnn_hidden_units,
                hidden_activations=self.config.dnn_activations,
                output_activation=None,
                dropout_rates=self.config.dnn_dropout,
                batch_norm=self.config.dnn_batch_norm,
            )
            output_dim = self.config.stacked_dnn_hidden_units[-1]
        if 'parallel' in self.config.model_structure:
            self.parallel_dnn = MLPLayer(
                input_dim=self.config.hidden_size * 2,
                output_dim=None,  # output hidden layer
                hidden_units=self.config.parallel_dnn_hidden_units,
                hidden_activations=self.config.dnn_activations,
                output_activation=None,
                dropout_rates=self.config.dnn_dropout,
                batch_norm=self.config.dnn_batch_norm,
            )
            output_dim = self.config.hidden_size * 2 + self.config.parallel_dnn_hidden_units[-1]
        if self.config.model_structure == 'stacked_parallel':
            output_dim = self.config.stacked_dnn_hidden_units[-1] + self.config.parallel_dnn_hidden_units[-1]
        if self.config.model_structure == 'crossnet_only':
            output_dim = self.config.hidden_size * 2

        self.prediction = nn.Linear(output_dim, 1)

    def predict(self, user_embeddings, item_embeddings) -> torch.Tensor:
        """
        @param user_embeddings: [B, D]  batch size, embedding size
        @param item_embeddings: [B, D]  batch size, embedding size
        @return: [B]  batch size
        """
        input_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)  # [batch_size, 2 * hidden_size]
        cross_output = self.cross_net(input_embeddings)
        if self.config.model_structure == 'crossnet_only':
            final_out = cross_output
        elif self.config.model_structure == 'stacked':
            final_out = self.stacked_dnn(cross_output)
        elif self.config.model_structure == 'parallel':
            dnn_out = self.parallel_dnn(cross_output)
            final_out = torch.cat([cross_output, dnn_out], dim=-1)
        else:
            final_out = torch.cat([self.stacked_dnn(cross_output), self.parallel_dnn(cross_output)], dim=-1)
        return self.prediction(final_out).flatten()
