from typing import Dict, Type, Union

from loader.class_hub import ClassHub
from loader.controller import Controller
from loader.mode.base_mode import BaseMode


class ModeHub:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.mode_class_hub = ClassHub.modes()
        self.mode_hub = dict()  # type: Dict[Union[str, Type[BaseMode]], BaseMode]

        for mode in self.mode_class_hub:
            mode_class = self.mode_class_hub(mode)  # type: Type[BaseMode]
            self.mode_hub[mode_class] = self.mode_hub[mode.lower()] = mode_class(
                legommender=self.controller.legommender,
                controller=self.controller,
                mode_hub=self,
            )

    def __call__(self, mode):
        return self.mode_hub[mode]
