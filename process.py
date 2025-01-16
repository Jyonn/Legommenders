from typing import Type

import pigmento

from pigmento import pnt
from unitok import UniTok
from unitok.utils.class_pool import ClassPool

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

    args = CommandInit(
        required_args=['data'],
        default_args=dict(
            regenerate=False,
            tokenizer=None,
            vocab=None,
        )
    ).parse()
    processor_class = get_processor(args.data)
    data_dir = DataInit.get(args.data)

    processor = processor_class(data_dir=data_dir)
    processor.load(regenerate=args.regenerate)

    if args.tokenizer is not None:
        if not args.vocab:
            raise ValueError('Tokenizer classname and vocabulary must be specified')

        tokenizer_params = dict()
        for key in args:
            if key.startswith('tokenizer_'):
                tokenizer_params[key[10:]] = args[key]

        with processor.item as ut:  # type: UniTok
            tokenizers = ClassPool.tokenizers(tokenizer_lib=args.lib or None)
            if args.tokenizer not in tokenizers:
                raise ValueError(f'Unknown tokenizer: {args.tokenizer}. Available tokenizers: {tokenizers.keys()}')
            tokenizer = tokenizers[args.tokenizer](vocab=args.vocab, **tokenizer_params)

        processor.add_item_tokenizer(tokenizer)
        processor.item.tokenize().save(processor.item_save_dir)
