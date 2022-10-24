from typing import Optional, Dict

import torch


class BaseBatch:
    def __init__(self, batch):
        from task.base_task import BaseTask

        self.append_info = batch['append_info']  # type: Dict[str, any]
        self.task = None  # type: Optional[BaseTask]
        self._registered_items = set()

        self.register('append_info', 'task')

    def register(self, *keys):
        self._registered_items.update(set(keys))

    def export(self):
        batch = dict()
        for key in self._registered_items:
            value = getattr(self, key)
            if isinstance(value, BaseBatch):
                value = value.export()
            batch[key] = value
        return batch


class SeqBatch(BaseBatch):
    def __init__(self, batch):
        super().__init__(batch=batch)

        self.input_ids = batch['input_ids']  # type: Optional[torch.Tensor]
        self.attention_mask = batch['attention_mask']  # type: Optional[torch.Tensor]
        self.segment_ids = batch['col_ids']  # type: Optional[torch.Tensor]
        self.col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        self.attr_ids = batch['attr_ids']  # type: Dict[str, Dict[str, torch.Tensor]]

        self.batch_size = int(self.input_ids.shape[0])
        self.seq_size = int(self.input_ids.shape[1])

        self.register('input_ids', 'attention_mask', 'segment_ids', 'col_mask', 'attr_ids')


class BartBatch(BaseBatch):
    batcher = SeqBatch

    def __init__(self, batch):
        super().__init__(batch=batch)

        batch['encoder']['append_info'] = batch['decoder']['append_info'] = batch['append_info']
        self.encoder = self.batcher(batch['encoder'])
        self.decoder = self.batcher(batch['decoder'])

        self.register('encoder', 'decoder')
