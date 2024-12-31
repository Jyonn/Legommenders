import abc

from transformers import AutoModelForCausalLM, LlamaForCausalLM

from embedder.base_embedder import BaseEmbedder


class LlamaEmbedder(BaseEmbedder, abc.ABC):
    def __init__(self, key):
        super().__init__(key)

        self.transformer = AutoModelForCausalLM.from_pretrained(self.key)  # type: LlamaForCausalLM

    def _get_embeddings(self):
        return self.transformer.model.embed_tokens


class Llama1Embedder(LlamaEmbedder):
    def __init__(self):
        super().__init__(key='huggyllama/llama-7b')


class Llama2Embedder(LlamaEmbedder):
    def __init__(self):
        super().__init__(key='meta-llama/Llama-2-7b-hf')


class Llama3Embedder(LlamaEmbedder):
    def __init__(self):
        super().__init__(key='meta-llama/Meta-Llama-3-8B')
