import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, BertForMaskedLM, BertConfig, BertModel, load_tf_weights_in_bert


class EmbeddingLoader:
    def __init__(self, vocab_type, path, frozen):
        self.type = vocab_type
        self.path = path
        self.embedding = None  # type: Optional[torch.Tensor]
        self.frozen = frozen

    @staticmethod
    def get_numpy_embedding(path):
        embedding = np.load(path)
        assert isinstance(embedding, np.ndarray)
        return torch.tensor(embedding, dtype=torch.float32)

    @staticmethod
    def get_bert_torch_embedding(path):
        bert_for_masked_lm = AutoModelForMaskedLM.from_pretrained(path)  # type: BertForMaskedLM
        bert = bert_for_masked_lm.bert
        return bert.embeddings.word_embeddings.weight

    @staticmethod
    def get_bert_tf_embedding(path):
        config = BertConfig.from_json_file(os.path.join(path, 'bert_config.json'))
        bert = BertModel(config)
        load_tf_weights_in_bert(bert, config, os.path.join(path, 'bert_model.ckpt.index'))
        return bert.embeddings.word_embeddings.weight

    def load(self):
        if hasattr(self, 'get_{}_embedding'.format(self.type)):
            getter = getattr(self, 'get_{}_embedding'.format(self.type))
            self.embedding = getter(self.path)
        return self
