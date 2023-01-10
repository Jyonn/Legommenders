import torch

from utils.iterating import Iterating
from utils.structure import Structure


class Reshaper(Iterating):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

    def custom_worker(self, x):
        if isinstance(x, torch.Tensor):
            if self.batch_size is None:
                self.batch_size = x.shape[0]
            else:
                assert self.batch_size == x.shape[0], 'Batch size mismatch'
            return x.view(-1, x.shape[-1])


class Recover(Iterating):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def custom_worker(self, x):
        if isinstance(x, torch.Tensor):
            return x.view(self.batch_size, -1, x.shape[-1])


class Shaper:
    def __init__(self):
        self.structure = None
        self.reshaper = Reshaper(batch_size=None)

    def transform(self, data: any):
        self.reshaper.batch_size = None
        self.structure = Structure(use_shape=True).analyse(data)
        return self.reshaper.worker(data)

    def recover(self, data: any):
        recover = Recover(batch_size=self.reshaper.batch_size)
        return recover.worker(data)


if __name__ == '__main__':
    shaper = Shaper()

    d = {
        "input_ids": {
            "title": torch.rand([64, 5, 23]),
            "cat": torch.rand([64, 5, 23]),
            "__cat_inputer_special_ids": torch.rand([64, 5, 23])
        },
        "attention_mask": torch.rand([64, 5, 23])
    }

    print(Structure().analyse_and_stringify(d))

    d = shaper.transform(d)

    print(Structure().analyse_and_stringify(d))

    d = torch.rand(64 * 5, 768)

    print(Structure().analyse_and_stringify(d))

    d = shaper.recover(d)

    print(Structure().analyse_and_stringify(d))
