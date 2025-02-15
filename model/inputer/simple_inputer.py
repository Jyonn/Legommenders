from typing import Optional, List, Dict

import torch
from unitok import Vocab

from loader.env import Env
from model.inputer.base_inputer import BaseInputer
from utils.slice_dict import SliceOrderedDict, SliceDict


class SimpleInputer(BaseInputer):
    output_single_sequence = False

    def get_vocabs(self) -> Optional[List[Vocab]]:
        return []

    @classmethod
    def pad(cls, l: list, max_len: int):
        # return padded list and mask
        return l + [Env.UNSET] * (max_len - len(l)), [1] * len(l) + [0] * (max_len - len(l))

    def sample_rebuilder(self, sample: dict):
        input_ids = dict()
        attention_mask = dict()

        for col in self.inputs:
            max_len = self.ut.meta.jobs[col].max_len
            if not max_len:
                sample[col] = [sample[col]]
                max_len = 1
            input_ids[col], attention_mask[col] = self.pad(sample[col], max_len)
            input_ids[col] = torch.tensor(input_ids[col])
            attention_mask[col] = torch.tensor(attention_mask[col])

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def get_mask(self, batched_samples: Dict[str, torch.Tensor]):
        return SliceDict(batched_samples['attention_mask'])

    def get_embeddings(
            self,
            batched_samples: Dict[str, torch.Tensor],
    ):
        input_embeddings = SliceOrderedDict()

        input_ids = batched_samples['input_ids']
        attention_mask = batched_samples['attention_mask']
        for col in input_ids:
            col_input = input_ids[col]  # batch_size, content_len
            vocab = self.ut.meta.jobs[col].tokenizer.vocab.name

            seq = col_input.to(Env.device)  # type: torch.Tensor
            # mask = (seq > Setting.UNSET).long()  # type: torch.Tensor
            mask = attention_mask[col].to(Env.device)
            seq *= mask

            embedding = self.embedding_hub(vocab, col_name=col)(seq)  # batch_size, content_len, embedding_dim
            mask = mask.unsqueeze(-1)  # batch_size, (content_len,) 1

            embedding *= mask  # batch_size, content_len, embedding_dim
            input_embeddings[col] = embedding

        return input_embeddings
