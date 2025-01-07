import os


class PathHub:
    def __init__(self, data_name, model_name, signature):
        self.data_name = data_name
        self.model_name = model_name
        self.signature = signature

        os.makedirs(self.checkpoint_base_dir, exist_ok=True)

        with open(self.log_path, 'w') as f:
            pass

    @property
    def checkpoint_base_dir(self):
        return os.path.join('checkpoints', self.data_name, self.model_name)

    @property
    def log_path(self):
        return os.path.join(self.checkpoint_base_dir, f'{self.signature}.log')

    @property
    def cfg_path(self):
        return os.path.join(self.checkpoint_base_dir, f'{self.signature}.json')

    @property
    def ckpt_path(self):
        return os.path.join(self.checkpoint_base_dir, f'{self.signature}.pt')

    @property
    def result_path(self):
        return os.path.join(self.checkpoint_base_dir, f'{self.signature}.csv')
