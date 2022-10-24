import json

from oba import Obj

from loader.global_loader import GlobalLoader
from loader.global_setting import Setting
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logger import Logger
from utils.printer import printer, Color, Printer


class Worker:
    def __init__(self, config):
        self.config = config
        Setting.device = self.get_device()

        self.print = printer[('MAIN', '·', Color.CYAN)]
        self.logging = Logger(self.config.data.store.log_path)
        Printer.logger = self.logging
        self.print(Obj.raw(self.config))

        self.global_loader = GlobalLoader(
            data=self.config.data,
            model=self.config.model,
            exp=self.config.exp
        )

        self.print(self.global_loader.a_depot[0])

    def get_device(self):
        cuda = self.config.cuda
        if cuda in [-1, False]:
            return 'cpu'
        if not cuda:
            return GPU.auto_choose(torch_format=True)
        return f"cuda:{cuda}"


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model', 'exp'],
        makedirs=[
            'data.store.save_dir',
        ]
    )
