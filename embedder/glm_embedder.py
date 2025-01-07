import abc

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedder.base_embedder import BaseEmbedder
from model.common.glm_interface import ChatGLMModel
from utils.config_init import ModelInit


class GLMEmbedder(BaseEmbedder, abc.ABC):
    transformer: ChatGLMModel

    def __init__(self):
        super().__init__()

        self.transformer = AutoModelForCausalLM.from_pretrained(ModelInit.get(str(self)), trust_remote_code=True)
        self.transformer: ChatGLMModel = self.transformer.transformer
        self.tokenizer = AutoTokenizer.from_pretrained(ModelInit.get(str(self)), trust_remote_code=True)

    def _get_embeddings(self):
        vocab_size = self.tokenizer.vocab_size
        embeddings = self.transformer.get_input_embeddings()
        embeddings = embeddings.weight[:vocab_size]
        return nn.Embedding.from_pretrained(embeddings)


class GLM4TH9BEmbedder(GLMEmbedder):
    pass
