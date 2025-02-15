import os.path
from typing import Type

import numpy as np
import pigmento
import yaml

from pigmento import pnt
from unitok import Vocab

from loader.class_hub import ClassHub
from processor.base_processor import BaseProcessor
from processor.recbench_processor import RecBenchProcessor
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
        required_args=['data', 'model'],
        default_args=dict()
    ).parse()
    processor_class = get_processor(args.data)
    data_dir = DataInit.get(args.data)

    processor = processor_class(data_dir=data_dir)
    if not isinstance(processor, RecBenchProcessor):
        raise ValueError('Only RecBenchProcessor is supported')

    processor.load()

    item_vocab = dict(zip(processor.item_df[processor.IID_COL], range(len(processor.item_df))))  # full vocab
    print(f'Loaded item vocab with size {len(item_vocab)}')

    # split data_dir (/path/to/recbench/data/<dataset>) to get dataset name
    dataset = data_dir.split(os.path.sep)[-1]
    recbench_dir = os.path.dirname(os.path.dirname(data_dir))

    embed_path = os.path.join(recbench_dir, 'export', dataset, f'{args.model}-embeds.npy')
    embeddings = np.load(embed_path)
    print(f'Loaded embeddings from {embed_path}, shape: {embeddings.shape}')

    if embeddings.shape[0] != len(item_vocab):
        raise ValueError(f'Embedding shape {embeddings.shape} does not match item vocab size {len(item_vocab)}')

    current_vocab = processor.item.meta.jobs[processor.IID_JOB].tokenizer.vocab  # type: Vocab
    current_embeddings = []
    for item in current_vocab:
        current_embeddings.append(embeddings[item_vocab[item]])
    current_embeddings = np.array(current_embeddings)

    embed_path = os.path.join(processor.save_dir, f'{args.model}-embeds.npy')
    np.save(embed_path, current_embeddings)

    embed_config = dict(
        embeddings=[
            dict(
                frozen=True,
                path=embed_path,
                col_name='item_id',
            )
        ],
        transformation='auto',
        transformation_dropout=0.1,
    )

    yaml_path = os.path.join('config', 'embed', f'{processor.get_name()}-{args.model}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(embed_config, f)

    print(f'Embedding configuration saved to {yaml_path}, please use `python trainer.py --embed {yaml_path}` to train')
