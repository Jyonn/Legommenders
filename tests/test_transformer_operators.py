import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel, GlmConfig, GlmModel, LlamaConfig, LlamaModel, OPTConfig, OPTModel

from loader.env import Env
from model.operators.bert_iisan_operator import BertBaseIISANOperator
from model.operators.bert_operator import BertBaseOperator
from model.operators.glm_iisan_operator import GLM4TH9BIISANOperator
from model.operators.glm_operator import GLM4TH9BOperator
from model.operators.llama_iisan_operator import Llama1IISANOperator
from model.operators.llama_operator import Llama1Operator
from model.operators.opt_iisan_operator import OPTBaseIISANOperator
from model.operators.opt_operator import OPTBaseOperator
from utils.config_init import ModelInit


class EmbeddingInputer:
    max_sequence_len = 5

    def __init__(self, **kwargs):
        pass

    def __call__(self, sample):
        return sample

    def get_embeddings(self, sample):
        return sample["embeddings"]

    def get_mask(self, sample):
        return sample["mask"]


class TransformerOperatorTest(unittest.TestCase):
    def test_cached_and_full_forward_match(self):
        configurations = (
            (BertBaseOperator, BertBaseIISANOperator, BertModel, BertConfig, {}, 16),
            (GLM4TH9BOperator, GLM4TH9BIISANOperator, GlmModel, GlmConfig,
             {"head_dim": 8, "num_key_value_heads": 2, "pad_token_id": 0}, 16),
            (Llama1Operator, Llama1IISANOperator, LlamaModel, LlamaConfig, {}, 16),
            (OPTBaseOperator, OPTBaseIISANOperator, OPTModel, OPTConfig,
             {"ffn_dim": 32, "word_embed_proj_dim": 8}, 8),
        )
        for operator_class, iisan_class, model_class, config_class, options, input_width in configurations:
            with self.subTest(model=model_class.__name__), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(2023)
                root = Path(directory)
                config = config_class(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=3, num_attention_heads=2, **options,
                )
                model_class(config).save_pretrained(root / "model")
                embeddings = torch.randn(2, 5, input_width)
                mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
                lego_config = SimpleNamespace(
                    item_ut=[dict(embeddings=x, mask=m) for x, m in zip(embeddings, mask)],
                    item_inputs=[], eh=None, item_page_size=2,
                )
                with (
                    patch.object(Env, "device", torch.device("cpu")),
                    patch.object(Env, "ph", SimpleNamespace(data_name=str(root))),
                    patch.object(ModelInit, "get", return_value=str(root / "model")),
                    patch.object(operator_class, "inputer_class", EmbeddingInputer),
                    patch.object(iisan_class, "inputer_class", EmbeddingInputer),
                ):
                    def make_operator(tune_from, use_lora=False):
                        config = operator_class.config_class(
                            hidden_size=8, input_dim=input_width, tune_from=tune_from,
                            use_lora=use_lora, lora_r=2, lora_alpha=4,
                        )
                        return operator_class(config=config, lego_config=lego_config)

                    full = make_operator(0).eval()
                    self.assertFalse(full.use_lm_cache())
                    full.cache([1, 2])
                    with torch.no_grad():
                        expected = full(embeddings, mask)

                    for boundary in (-2, -1):
                        cached = make_operator(boundary).eval()
                        cached.linear.load_state_dict(full.linear.state_dict())
                        cached.additive_attention.load_state_dict(full.additive_attention.state_dict())
                        with torch.no_grad():
                            actual = cached(torch.arange(2))
                        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)

                    for boundary in (0, -2):
                        tuned = make_operator(boundary, use_lora=True)
                        inputs = torch.arange(2) if tuned.use_lm_cache() else embeddings
                        tuned(inputs, mask).square().mean().backward()
                        self.assertTrue(any(
                            param.grad is not None
                            for name, param in tuned.named_parameters()
                            if "lora_" in name
                        ))

                    config = iisan_class.config_class(hidden_size=8, input_dim=input_width)
                    iisan = iisan_class(config=config, lego_config=lego_config)
                    output = iisan(torch.arange(2))
                    self.assertEqual(output.shape, (2, 8))
                    self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
