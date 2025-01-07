import torch.utils.checkpoint
from torch import nn
from typing import Optional, Tuple

from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"
    num_layers: int
    padded_vocab_size: int
    hidden_size: int
    ffn_hidden_size: int
    kv_channels: int
    num_attention_heads: int
    seq_length: int
    hidden_dropout: float
    classifier_dropout: float
    attention_dropout: float
    layernorm_epsilon: float
    rmsnorm: bool
    apply_residual_connection_post_layernorm: bool
    post_layer_norm: bool
    add_bias_linear: bool
    add_qkv_bias: bool
    bias_dropout_fusion: bool
    multi_query_attention: bool
    multi_query_group_num: int
    rope_ratio: int
    apply_query_key_layer_scaling: bool
    attention_softmax_in_fp32: bool
    fp32_residual_connection: bool


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class RotaryEmbedding(nn.Module):
    dim: int
    inv_freq: torch.Tensor
    original_impl: bool
    rope_ratio: int

    def forward(self, max_seq_len, offset=0):
        raise NotImplementedError


class CoreAttention(torch.nn.Module):
    config: ChatGLMConfig
    apply_query_key_layer_scaling: bool
    attention_softmax_in_fp32: bool
    layer_number: int
    is_causal: bool
    hidden_size_per_partition: int
    hidden_size_per_attention_head: int
    num_attention_heads_per_partition: int
    norm_factor: float
    coeff: Optional[int]
    attention_dropout: torch.nn.Dropout

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        raise NotImplementedError


class SelfAttention(torch.nn.Module):
    layer_number: int
    projection_size: int
    hidden_size_per_attention_head: int
    num_attention_heads_per_partition: int
    multi_query_attention: bool
    qkv_hidden_size: int
    num_multi_query_groups_per_partition: int
    query_key_value: nn.Linear
    core_attention: CoreAttention
    dense: nn.Linear

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        pass


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    add_bias: bool
    dense_h_to_4h: nn.Linear
    activation_func: nn.Module
    dense_4h_to_h: nn.Linear

    def forward(self, hidden_states):
        raise NotImplementedError


class GLMBlock(torch.nn.Module):
    layer_number: int
    apply_residual_connection_post_layernorm: bool
    fp32_residual_connection: bool
    input_layernorm: nn.Module
    self_attention: SelfAttention
    hidden_dropout: float
    post_attention_layernorm: nn.Module
    mlp: MLP

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        raise NotImplementedError


class GLMTransformer(torch.nn.Module):
    fp32_residual_connection: bool
    post_layer_norm: bool
    num_layers: int
    layers: torch.nn.ModuleList
    final_layernorm: nn.Module
    gradient_checkpointing: bool

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        raise NotImplementedError


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        raise NotImplementedError

    def get_position_ids(self, input_ids, device):
        raise NotImplementedError


class Embedding(torch.nn.Module):
    hidden_size: int
    word_embeddings: nn.Embedding
    fp32_residual_connection: bool


class ChatGLMModel(ChatGLMPreTrainedModel):
    embedding: Embedding
    num_layers: int
    multi_query_group_num: int
    kv_channels: int
    seq_length: int
    rotary_pos_emb: RotaryEmbedding
    encoder: GLMTransformer
    output_layer: nn.Linear
    config: ChatGLMConfig

    def get_input_embeddings(self):
        raise NotImplementedError

    def set_input_embeddings(self, value: any):
        raise NotImplementedError

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        raise NotImplementedError


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    transformer: ChatGLMModel
    max_sequence_length: int
    config: ChatGLMConfig
