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

from model.common.mlp_layer import MLPLayer
from model.predictors.base_predictor import BasePredictorConfig, BasePredictor


class DNNPredictorConfig(BasePredictorConfig):
    def __init__(
            self,
            dnn_hidden_units,
            dnn_activations,
            dnn_dropout,
            dnn_batch_norm,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activations = dnn_activations
        self.dnn_dropout = dnn_dropout
        self.dnn_batch_norm = dnn_batch_norm


class DNNPredictor(BasePredictor):
    config_class = DNNPredictorConfig
    config: DNNPredictorConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mlp = MLPLayer(
            input_dim=self.config.hidden_size * 2,
            output_dim=1,  # output hidden layer
            hidden_units=self.config.dnn_hidden_units,
            hidden_activations=self.config.dnn_activations,
            output_activation=None,
            dropout_rates=self.config.dnn_dropout,
            batch_norm=self.config.dnn_batch_norm,
            use_bias=True
        )

    def predict(self, user_embeddings, item_embeddings) -> torch.Tensor:
        """
        @param user_embeddings: [B, D]  batch size, embedding size
        @param item_embeddings: [B, D]  batch size, embedding size
        @return: [B]  batch size
        """
        input_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)  # [batch_size, 2 * hidden_size]
        final_out = self.mlp(input_embeddings)
        return final_out.flatten()
