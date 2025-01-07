import abc

from transformers import AutoModelForCausalLM, LlamaForCausalLM

from embedder.base_embedder import BaseEmbedder
from utils.config_init import ModelInit


class LlamaEmbedder(BaseEmbedder, abc.ABC):
    def __init__(self):
        super().__init__()

        self.transformer = AutoModelForCausalLM.from_pretrained(ModelInit.get(str(self)))  # type: LlamaForCausalLM

    def _get_embeddings(self):
        return self.transformer.model.embed_tokens


class Llama1Embedder(LlamaEmbedder):
    pass


class Llama2Embedder(LlamaEmbedder):
    pass


class Llama3Embedder(LlamaEmbedder):
    pass
