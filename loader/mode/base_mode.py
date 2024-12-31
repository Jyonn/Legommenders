from loader.manager import Manager


class BaseMode:
    load_model = False

    def __init__(
            self,
            manager: Manager,
            mode_hub,
    ):
        self.manager: Manager = manager
        self.legommender = self.manager.legommender
        self.mode_hub = mode_hub

    def work(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.work(*args, **kwargs)
