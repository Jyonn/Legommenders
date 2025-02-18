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
from model.predictors.base_predictor import BasePredictor
from model.predictors.dcn_predictor import DCNPredictorConfig


class GDCNPredictorConfig(DCNPredictorConfig):
    def __init__(
            self,
            sequential_mode: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.sequential_mode = sequential_mode


class GateCrossLayer(nn.Module):
    #  The core structure： gated cross layer.
    def __init__(self, input_dim, cross_layers=3):
        super().__init__()

        self.cross_layers = cross_layers

        self.w = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cross_layers)
        ])
        self.wg = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=False) for _ in range(cross_layers)
        ])

        self.b = nn.ParameterList([nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cross_layers)])

        for i in range(cross_layers):
            nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for i in range(self.cross_layers):
            xw = self.w[i](x)  # Feature Crossing
            xg = self.activation(self.wg[i](x))  # Information Gate
            x = x0 * (xw + self.b[i]) * xg + x
        return x


class GDCNPredictor(BasePredictor):
    config_class = GDCNPredictorConfig
    config: GDCNPredictorConfig

    def __init__(self, **kwargs):
        super(GDCNPredictor, self).__init__(**kwargs)

        output_dim = 1 if self.config.sequential_mode else None
        self.dnn = MLPLayer(
            input_dim=self.config.hidden_size * 2,
            output_dim=output_dim,
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True,
        )

        self.cross_net = GateCrossLayer(self.config.hidden_size * 2, self.config.cross_num)

        if not self.config.sequential_mode:
            output_dim = self.config.hidden_size * 2 + self.config.dnn_hidden_units[-1]
            self.prediction = nn.Linear(output_dim, 1)

    def predict(self, user_embeddings, item_embeddings):
        input_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)  # [batch_size, 2 * hidden_size]
        cross_output = self.cross_net(input_embeddings)

        if self.config.sequential_mode:
            dnn_output = self.dnn(cross_output)
            return dnn_output.flatten()

        dnn_output = self.dnn(input_embeddings)
        final_out = torch.cat([cross_output, dnn_output], dim=-1)
        return self.prediction(final_out).flatten()
