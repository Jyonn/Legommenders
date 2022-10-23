from utils.config_init import ConfigInit
from utils.gpu import GPU


class Worker:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_device(cuda):
        if cuda in [-1, False]:
            return 'cpu'
        if cuda is None:
            return GPU.auto_choose(torch_format=True)
        return "cuda:{}".format(cuda)


if __name__ == '__main__':
    config = ConfigInit(
        required_args=['data', 'model', 'exp'],
        makedirs=[
            'data.store.save_dir',
        ]
    )
