from typing import Type

import pigmento

from pigmento import pnt

from loader.class_hub import ClassHub
from processor.base_processor import BaseProcessor
from utils.config_init import CommandInit, DataInit


def get_processor(name) -> Type[BaseProcessor]:
    processor_hub = ClassHub.processors()
    name = name.upper()

    if name not in processor_hub:
        raise ValueError(f'Unknown processor: {name}')
    return processor_hub[name]


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pigmento.add_dynamic_color_plugin()
    pnt.set_display_mode(
        display_method_name=False,
        display_class_name=True,
        use_instance_class=True,
    )

    configuration = CommandInit(
        required_args=['data'],
        default_args=dict(regenerate=False)
    ).parse()
    processor_class = get_processor(configuration.data)
    data_dir = DataInit.get(configuration.data)

    processor = processor_class(data_dir=data_dir)
    processor.load(regenerate=configuration.regenerate)
