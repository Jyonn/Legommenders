import os

import numpy as np
import torch
from pigmento import pnt
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from loader.data_set import DataSet
from loader.env import Env
from loader.pager.iisan_layer_pager import IISANLayerPager
from model.inputer.concat_inputer import ConcatInputer
from model.operators.base_operator import BaseOperatorConfig
from model.operators.lm_operator import LMOperator
from utils import bars
from utils.config_init import ModelInit


class IISANOperatorConfig(BaseOperatorConfig):
    def __init__(
            self,
            global_proj_size: int = None,
            local_proj_size: int = None,
            layer_selection_step: int = 1,
            **kwargs
    ):
        super().__init__(**kwargs)

        if global_proj_size is not None and local_proj_size is not None and global_proj_size != local_proj_size:
            raise ValueError('global_proj_size and local_proj_size must be equal if both are set')
        self.global_proj_size = global_proj_size
        self.local_proj_size = local_proj_size
        self.layer_selection_step = layer_selection_step


class SANBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(SANBlock, self).__init__()
        self.fc_up = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc_down = nn.Linear(embedding_dim * 2, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, hidden_states):
        x = self.fc_up(hidden_states)
        x = torch.relu(x)
        x = self.fc_down(x)
        return self.norm(x + hidden_states)


class IISANOperator(LMOperator):
    config_class = IISANOperatorConfig
    inputer_class = ConcatInputer
    inputer: ConcatInputer
    config: IISANOperatorConfig
    dtype = torch.float32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = AutoModel.from_pretrained(self.transformer_key, trust_remote_code=True)
        self.num_hidden_layers = self.get_layer_nums()
        self.selected_layers = self.get_selected_layers()
        self.num_selected_layers = len(self.selected_layers)

        os.makedirs(self._cache_base_dir, exist_ok=True)

        self.san_block_input_dim = self.config.input_dim

        self.global_projection = None
        if self.config.global_proj_size is not None:
            self.san_block_input_dim = self.config.global_proj_size
            self.global_projection = nn.Linear(self.config.input_dim, self.config.global_proj_size, bias=False)

        self.local_projections = [None] * self.num_selected_layers
        if self.config.local_proj_size is not None:
            self.local_projections = nn.ModuleList([
                nn.Linear(self.config.input_dim, self.config.local_proj_size, bias=False)
                for _ in range(self.num_selected_layers)
            ])
            self.san_block_input_dim = self.config.local_proj_size

        self.san_blocks = nn.ModuleList([SANBlock(self.san_block_input_dim) for _ in range(self.num_selected_layers - 1)])
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(self.num_selected_layers - 1)
        ])  # gating factors

        self.linear = nn.Linear(self.san_block_input_dim, self.config.hidden_size)

        self.hidden_states = None  # [N, H, D]

    def get_selected_layers(self):
        if self.config.layer_selection_step > self.num_hidden_layers:
            raise ValueError(f'layer_selection_step {self.config.layer_selection_step} is larger than num_hidden_layers {self.num_hidden_layers}')
        selected_layers = list(range(0, self.num_hidden_layers, self.config.layer_selection_step))
        margin = self.num_hidden_layers - selected_layers[-1] - 1
        selected_layers = list(map(lambda x: x + margin, selected_layers))
        selected_layer_str = ', '.join(map(str, selected_layers))
        pnt(f'selected_layers: {selected_layer_str}')
        return selected_layers

    def use_lm_cache(self):
        return True

    @property
    def transformer_key(self):
        return ModelInit.get(self.operator_name.replace('iisan', ''))

    def get_layer_nums(self):
        return int(self.transformer.config.num_hidden_layers)

    def __str__(self):
        return self.operator_name

    def _load_hidden_states(self):
        if self._cache_exists():
            pnt(f'loading cached hidden states from {self._get_cache_path()}')
        else:
            pnt(f'caching item hidden states at runtime')
            self.cache()
        hidden_states = np.load(self._get_cache_path())
        selected_hidden_states = []
        for layer in self.selected_layers:
            selected_hidden_states.append(hidden_states[:, layer, :])
        selected_hidden_states = np.stack(selected_hidden_states, axis=1)  # [N, H, D]

        self.hidden_states = torch.from_numpy(selected_hidden_states)
        pnt(f'hidden_weights.shape: {self.hidden_states.shape}')  # [N, H, D]

        nan_mask = torch.isnan(self.hidden_states).any(dim=-1)  # [N, L]
        self.hidden_states[nan_mask] = torch.rand_like(self.hidden_states[nan_mask]).to(self.hidden_states.dtype)

        self.transformer = None
        torch.cuda.empty_cache()

    @property
    def _cache_base_dir(self):
        return os.path.join('cache', Env.ph.data_name, self.operator_name)

    def _cache_exists(self):
        return os.path.exists(self._get_cache_path())

    def _get_cache_path(self):
        return os.path.join(self._cache_base_dir, f'states.npy')

    def get_pretrained_parameter_names(self):
        return ['transformer']

    def get_hidden_states(
            self,
            inputs_embeds,
            attention_mask,
    ):
        output: BaseModelOutputWithPast = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        states = output.hidden_states  # [B, L+1, S, D], is a list of tensors, not tensor
        states = states[1:]  # remove the embedding layer
        # calculate mean pooling based on mask
        states = torch.stack(states, dim=1)  # [B, L, S, D]
        mask = attention_mask[:, None, :, None].to(states.dtype)
        masked_states = states * mask
        states = masked_states.sum(dim=2) / mask.sum(dim=2)
        return states

    def forward(self, embeddings, mask=None, **kwargs):
        indices = embeddings.cpu()  # [B]
        hidden_states = self.hidden_states[indices].to(Env.device)  # [B, H, D]

        if self.global_projection is not None:
            hidden_states = self.global_projection(hidden_states)

        current_state = hidden_states[:, 0, :]
        if self.local_projections[0] is not None:
            current_state = self.local_projections[0](current_state)

        for i in range(self.num_selected_layers - 1):
            hidden_state = hidden_states[:, i + 1, :]
            if self.local_projections[i + 1] is not None:
                hidden_state = self.local_projections[i + 1](hidden_state)
            gate = torch.sigmoid(self.gates[i])  # μ in (0,1)
            fusion = gate * current_state + (1 - gate) * hidden_state
            current_state = self.san_blocks[i](fusion)

        return self.linear(current_state)

    @property
    def cache_hidden_size(self):
        return self.config.input_dim

    def cache(self):
        dataset = DataSet(ut=self.lego_config.item_ut)
        contents = []
        for sample in bars.DescBar(desc='Caching Item Content')(dataset):
            contents.append(self.inputer(sample))

        self.to(Env.device)

        pager = IISANLayerPager(
            inputer=self.inputer,
            num_layers=self.num_hidden_layers,
            hidden_size=self.cache_hidden_size,
            contents=contents,
            model=self.get_hidden_states,
            page_size=self.lego_config.item_page_size,
        )

        pager.run()

        np.save(self._get_cache_path(), pager.final_states.float().numpy())
        pnt(f'caches saved to {self._cache_base_dir}')


