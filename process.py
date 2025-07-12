"""
process_cli.py

Command-line entry point that prepares **raw data** for an experiment.

The script performs three high-level tasks:

1. Receive user arguments (dataset name, whether to re-generate cached
   files, optional tokenizer information, …).
2. Look-up and instantiate the correct `BaseProcessor` subclass that
   knows how to turn the selected raw corpus into the intermediate
   format required by the training pipeline.
3. (Optional) Attach a user-specified *tokenizer* to the item field and
   persist the updated dataset.

The actual heavy lifting (reading CSVs, normalising features, building
vocabularies, …) is delegated to the individual `Processor` classes
so that this file remains a thin orchestration layer.
"""

from __future__ import annotations

from typing import Type, Dict, Any

import pigmento
from pigmento import pnt

from unitok import UniTok
from unitok.utils.class_pool import ClassPool

# Project utilities / abstractions
from loader.class_hub import ClassHub
from processor.base_processor import BaseProcessor
from utils.config_init import CommandInit, DataInit


# ---------------------------------------------------------------------- #
# Helper functions                                                       #
# ---------------------------------------------------------------------- #
def get_processor(name: str) -> Type[BaseProcessor]:
    """
    Fetch the proper `BaseProcessor` subclass for the given dataset name.

    Parameters
    ----------
    name : str
        Case-insensitive key that identifies the dataset.

    Returns
    -------
    Type[BaseProcessor]
        Concrete class that can handle the dataset.

    Raises
    ------
    ValueError
        If no matching processor is registered.
    """
    processor_hub = ClassHub.processors()   # {'MOVIELENS': MovielensProcessor, …}
    name = name.upper()

    if name not in processor_hub:
        raise ValueError(f'Unknown processor: {name}')
    return processor_hub[name]


# ---------------------------------------------------------------------- #
# Main routine                                                           #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    # -------- Pretty console logging ---------------------------------- #
    pigmento.add_time_prefix()
    pigmento.add_dynamic_color_plugin()
    pnt.set_display_mode(
        display_method_name=False,
        display_class_name=True,
        use_instance_class=True,
    )

    # -------- Argument parsing ---------------------------------------- #
    # We only demand a `--data <DATASET_NAME>` argument. Everything else
    # falls back to default values.
    args = CommandInit(
        required_args=['data'],
        default_args=dict(
            regenerate=False,      # force re-processing even if cache exists
            tokenizer=None,        # which tokenizer to use for the item field
            vocab=None,            # vocabulary path needed by most tokenizers
        )
    ).parse()

    # -------- Processor bootstrap ------------------------------------- #
    processor_class = get_processor(args.data)
    data_dir = DataInit.get(args.data)       # maps dataset name -> local folder

    processor = processor_class(data_dir=data_dir)
    # Either loads cached output or runs the `process()` routine again.
    processor.load(regenerate=args.regenerate)

    # -------- Optional: add tokenizer for item text ------------------- #
    # A tokenizer can be attached *after* the main processing step.
    if args.tokenizer is not None:
        if not args.vocab:
            raise ValueError(
                'Both --tokenizer <CLASSNAME> and --vocab <VOCAB_PATH> '
                'must be supplied.'
            )

        # Collect kwargs that should be forwarded to the tokenizer ctor.
        tokenizer_params: Dict[str, Any] = {}
        for key in args:
            # CLI flags are expected to look like `--tokenizer_max_len`
            if key.startswith('tokenizer_'):
                tokenizer_params[key[10:]] = args[key]

        # Resolve the tokenizer class dynamically.
        with processor.item as ut:  # type: UniTok
            tokenizers = ClassPool.tokenizers(tokenizer_lib=args.lib or None)
            if args.tokenizer not in tokenizers:
                raise ValueError(
                    f'Unknown tokenizer: {args.tokenizer}. '
                    f'Available tokenizers: {list(tokenizers.keys())}'
                )
            tokenizer_cls = tokenizers[args.tokenizer]

            # The UniTok dataset is context-manager aware – we open it here
            # so the files are closed automatically afterwards.
            tokenizer = tokenizer_cls(vocab=args.vocab, **tokenizer_params)

        # Register the tokenizer with the processor and write changes to disk.
        processor.add_item_tokenizer(tokenizer)
        processor.item.tokenize().save(processor.item_save_dir)

        pnt(f'Successfully tokenized items using {args.tokenizer} '
            f'and saved output to {processor.item_save_dir}')
