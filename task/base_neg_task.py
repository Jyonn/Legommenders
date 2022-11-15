import random
from abc import ABC

from set.base_dataset import BaseDataset
from task.base_task import BaseTask


class BaseNegTask(BaseTask, ABC):
    def __init__(
            self,
            dataset: BaseDataset,
            neg_count=4,
            neg_col='neg',
            **kwargs
    ):
        super().__init__(dataset, **kwargs)
        self.neg_count = neg_count
        self.neg_col = neg_col

    def negative_sampling(self, sample: dict, sample_size):
        neg_samples = []
        if not self.is_testing:
            rand_neg = self.neg_count
            neg_samples = []
            if self.neg_col and self.neg_col in sample['append']:
                true_negs = sample['append'][self.neg_col]
                rand_neg = max(self.neg_count - len(true_negs), 0)
                neg_samples = random.choices(true_negs, k=min(self.neg_count, len(true_negs)))
            neg_samples += [random.randint(0, sample_size - 1) for _ in range(rand_neg)]
        if self.neg_col and self.neg_col in sample['append']:
            del sample['append'][self.neg_col]
        return neg_samples
