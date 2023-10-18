from loader.controller import Controller
from model.legommender import Legommender


class BaseMode:
    load_model = False

    def __init__(
            self,
            legommender: Legommender,
            controller: Controller,
            mode_hub,
    ):
        self.legommender = legommender
        self.controller = controller
        self.mode_hub = mode_hub

    def work(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.work(*args, **kwargs)
