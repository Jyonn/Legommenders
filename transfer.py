"""
recbench_embed_converter.py

Utility
-------
Take *global* item embeddings produced by the RecBench toolkit and
convert them into a **dataset-specific** NumPy array that matches the
internal `UniTok` vocabulary created by `RecBenchProcessor`.

The script also generates a ready-to-use YAML file for the `--embed`
flag accepted by `trainer.py`.

Workflow
--------
1. Parse CLI arguments (`--data  <DATASET>`  and  `--model <MODEL_ID>`).
2. Load the correct `RecBenchProcessor` and its cached data frames.
3. Read the *full* item embedding matrix exported by RecBench
   (`export/<dataset>/<model>-embeds.npy`).
4. Re-order the rows so they align with the *current* `uni_tok.Vocab`
   and store the resulting matrix alongside the processed dataset.
5. Create a small YAML file that points to the new `.npy` file so the
   training pipeline can seamlessly pick it up.

Example
-------
python recbench_embed_converter.py --data beauty --model mlp
"""
from __future__ import annotations

import os
from typing import Type, Dict

import numpy as np
import pigmento
import yaml
from pigmento import pnt
from unitok import Vocab

# Project-internal imports
from loader.class_hub import ClassHub
from processor.base_processor import BaseProcessor
from processor.recbench_processor import RecBenchProcessor
from utils import io
from utils.config_init import CommandInit, DataInit


# ---------------------------------------------------------------------- #
# Helper: resolve processor class                                        #
# ---------------------------------------------------------------------- #
def get_processor(name: str) -> Type[BaseProcessor]:
    """
    Return the registered `BaseProcessor` subclass for the given dataset.

    Parameters
    ----------
    name : str
        Human readable dataset identifier (case-insensitive).

    Raises
    ------
    ValueError
        If the dataset is unknown.
    """
    processor_hub = ClassHub.processors()
    name = name.upper()

    if name not in processor_hub:
        raise ValueError(f'Unknown processor: {name}')
    return processor_hub[name]


# ---------------------------------------------------------------------- #
# Main routine                                                           #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    # -------- Colourful logging --------------------------------------- #
    pigmento.add_time_prefix()
    pigmento.add_dynamic_color_plugin()
    pnt.set_display_mode(
        display_method_name=False,
        display_class_name=True,
        use_instance_class=True,
    )

    # -------- Argument parsing ---------------------------------------- #
    args = CommandInit(
        required_args=['data', 'model'],
        default_args=dict()     # nothing extra needed
    ).parse()

    # -------- Processor bootstrap ------------------------------------- #
    processor_class = get_processor(args.data)
    data_dir = DataInit.get(args.data)

    processor = processor_class(data_dir=data_dir)
    if not isinstance(processor, RecBenchProcessor):
        raise ValueError('Only RecBenchProcessor is supported for this script.')

    processor.load()   # ensures `processor.item_df` etc. are available

    # ------------------------------------------------------------------ #
    # 1. Build *global* item → index mapping (RecBench vocabulary)       #
    # ------------------------------------------------------------------ #
    # RecBench stores the original `item_id` in the left-most column.
    item_vocab: Dict[str, int] = dict(
        zip(processor.item_df[processor.IID_COL], range(len(processor.item_df)))
    )
    pnt(f'Loaded item vocab with size {len(item_vocab)}')

    # ------------------------------------------------------------------ #
    # 2. Load exported embeddings                                        #
    # ------------------------------------------------------------------ #
    dataset = os.path.basename(data_dir)                    # name of the dataset folder
    recbench_dir = os.path.dirname(os.path.dirname(data_dir))
    embed_path = os.path.join(recbench_dir, 'export', dataset, f'{args.model}-embeds.npy')

    embeddings = np.load(embed_path)
    pnt(f'Loaded embeddings from {embed_path}, shape: {embeddings.shape}')

    if embeddings.shape[0] != len(item_vocab):
        raise ValueError(
            f'Embedding shape {embeddings.shape} does not match '
            f'item vocab size {len(item_vocab)}'
        )

    # ------------------------------------------------------------------ #
    # 3. Re-order embeddings to match UniTok vocabulary                  #
    # ------------------------------------------------------------------ #
    current_vocab: Vocab = processor.item.meta.features[processor.IID_FEAT].tokenizer.vocab
    current_embeddings = np.vstack([
        embeddings[item_vocab[item]] for item in current_vocab
    ])
    pnt(f'Re-ordered embeddings to match UniTok vocab, '
        f'new shape: {current_embeddings.shape}')

    # Persist matrix next to processed dataset so it can be distributed
    # together with the cached data.
    embed_path = os.path.join(processor.save_dir, f'{args.model}-embeds.npy')
    np.save(embed_path, current_embeddings)
    pnt(f'Saved re-ordered embeddings to {embed_path}')

    # ------------------------------------------------------------------ #
    # 4. Generate YAML configuration snippet                             #
    # ------------------------------------------------------------------ #
    embed_config = dict(
        embeddings=[
            dict(
                frozen=True,           # do NOT fine-tune RecBench vectors
                path=embed_path,
                col_name='item_id',    # mapping column inside UniTok dataset
            )
        ],
        transformation='auto',         # let the framework pick a suitable
        transformation_dropout=0.1,    # dropout before feeding to the model
    )

    # Store the YAML file under `config/embed/<dataset>-<model>.yaml`
    yaml_path = os.path.join(
        'config', 'embed', f'{processor.get_name()}-{args.model}.yaml'
    )
    io.yaml_save(embed_config, yaml_path)
    pnt(f'Embedding configuration saved to {yaml_path}')
    pnt(f'Use it via:  python trainer.py --embed {yaml_path}')
