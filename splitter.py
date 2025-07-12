"""
splitter.py

Layer-wise *pre-computation* for language-model based item encoders
==================================================================

Motivation
----------
Fine-tuning a large Transformer encoder for every training iteration can
become prohibitively expensive.  If we are only interested in *selected*
hidden layers (e.g. the output of layer 0, 6 and 11), we can compute
those representations **once** and store them on disk.  During training
the model simply looks up the cached vectors – no more forward pass
through the LM required.

This helper script automates exactly that workflow:

1. Initialize the full experiment stack by re-using `BaseLego`
   (datasets, model, device, …).
2. Assert that the model’s *item operator* is an instance of
   `OnceOperator`, i.e. an operator that supports a `.cache()` method
   for storing layer outputs.
3. Parse the user supplied `--layers` string, convert negative indices
   (e.g. `-1` → *last* layer) and trigger the caching routine.

CLI example
-----------
python splitter.py \
    --data  movielens \
    --model config/model/bert_recommender.yaml \
    --embed config/embed/bert.yaml \
    --layers 0+3+11

The cached files are written to the location chosen by the `OnceOperator`
implementation (usually somewhere within the dataset folder).

Notes
-----
• A *pre-trained embedding* configuration **must** be provided via
  `--embed`, otherwise there is no LM to run.

• The script is intentionally minimal – if you need more control (e.g.
  mixed precision, shard-wise processing, …) extend the `Splitter`
  class and override `run()`.
"""

from __future__ import annotations

from typing import List

from base_lego import BaseLego
from model.operators.once_operator import OnceOperator
from utils.config_init import CommandInit


class Splitter(BaseLego):
    """
    Sub-class of `BaseLego` that only implements the `run()` hook.
    """

    # ------------------------------------------------------------------ #
    # Main logic                                                         #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        1. Validate that the item operator supports caching.
        2. Convert the `--layers` CLI argument from *string* to *list[int]*.
        3. Call `.cache()` on the operator so that layer outputs are
           pre-computed and saved to disk.
        """
        # ------------------------------------------------------------------ #
        # 1. Safety checks                                                   #
        # ------------------------------------------------------------------ #
        item_op = self.legommender.item_op
        if not isinstance(item_op, OnceOperator):
            raise ValueError('Item encoder is not a `OnceOperator` – '
                             'layer splitting is therefore not supported.')

        if not self.embed.embeddings:
            raise ValueError(
                'Please specify *pre-trained embedding* configurations '
                '(`--embed <YAML>`) when using LM layer split.'
            )

        # ------------------------------------------------------------------ #
        # 2. Parse the `--layers` argument                                    #
        # ------------------------------------------------------------------ #
        # Accept syntax like "0+6+11" or "-1+2".
        user_layers: List[int] = list(map(int, self.config.layers.split('+')))

        # Convert negative indices (-1 == last layer, ‑2 == penultimate, …)
        num_layers = item_op.num_hidden_layers
        layers: List[int] = [
            l if l >= 0 else l + num_layers       # negative -> wrap around
            for l in user_layers
        ]

        # ------------------------------------------------------------------ #
        # 3. Kick off caching                                                 #
        # ------------------------------------------------------------------ #
        # The `OnceOperator` will take care of batching, device placement
        # and writing the resulting `.npy` (or similar) files.
        item_op.cache(layers)


# ---------------------------------------------------------------------- #
# CLI entry point                                                        #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model', 'embed', 'layers'],
        default_args=dict(
            exp='config/exp/default.yaml',
            # ---- arguments that are required by BaseLego but not used  #
            hidden_size=256,
            batch_size=64,
        ),
    ).parse()

    splitter = Splitter(config=configuration)
    splitter.run()
