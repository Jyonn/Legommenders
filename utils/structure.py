import json

import torch

from utils.iterating import Iterating


class TensorShape:
    def __init__(self, shape, dtype):
        self.shape = list(shape)
        self.dtype = dtype

    def __str__(self):
        return f'tensor({self.shape}, dtype={self.dtype})'

    def __repr__(self):
        return self.__str__()


class ListShape:
    def __init__(self, data):
        self.shape = []
        while isinstance(data, list):
            self.shape.append(len(data))
            data = data[0]

    def __str__(self):
        return f'list({self.shape})'

    def __repr__(self):
        return self.__str__()


class Structure(Iterating):
    def __init__(self, use_shape=False):
        super().__init__()
        self.use_shape = use_shape

    def custom_worker(self, x):
        if isinstance(x, torch.Tensor):
            if self.use_shape:
                return TensorShape(x.shape, x.dtype)
            return f'tensor({list(x.shape)}, dtype={x.dtype})'
        elif isinstance(x, list):
            shape = ListShape(x)
            if self.use_shape:
                return shape
            return str(shape)
        else:
            return type(x).__name__

    def worker(self, x):
        if isinstance(x, dict):
            return self.worker_dict(x)
        return self.custom_worker(x)

    def analyse(self, x):
        return self.worker(x)

    def analyse_and_stringify(self, x):
        assert not self.use_shape, 'Cannot stringify shape'
        structure = self.analyse(x)
        return json.dumps(structure, indent=4)


if __name__ == '__main__':
    # a = dict(
    #     x=torch.rand(3, 5, 6),
    #     y=dict(
    #         z=torch.rand(3, 6),
    #         k=[torch.rand(3, 2, 6), torch.rand(3)]
    #     )
    # )
    #

    a = dict(
        x=torch.rand(3, 5, 6).tolist(),
        y=dict(
            z=torch.rand(3, 6).tolist(),
            k=[torch.rand(3, 2, 6).tolist()]
        )
    )

    print(Structure(use_shape=True).analyse(a))
