"""
model_sizer.py

Utility script that reports the number of *trainable* parameters of a
model configuration.

Why does this live as a subclass of `BaseLego`?
------------------------------------------------
`BaseLego` already takes care of
    • reading and validating the YAML/CLI configuration,
    • building the data-pipe (`Manager`),
    • instantiating the actual model (`legommender`),
    • selecting the correct CUDA / CPU device, …
Hence we can simply plug our custom logic into the `run()` method and
reuse all of the heavy lifting done in the parent class.

Typical usage
-------------
$ python model_sizer.py --data movielens \
                        --model config/model/your_model.yaml \
                        --hidden_size 512

The script prints every *trainable* parameter tensor together with its
shape and finally the total parameter count (in millions).

Notes
-----
The parameter count is restricted to tensors that have
`requires_grad == True`.  Frozen embeddings or pre-trained encoders that
are excluded from fine-tuning are therefore not part of the final number.
"""

from __future__ import annotations

from typing import List, Tuple

from pigmento import pnt

from base_lego import BaseLego
from utils.config_init import CommandInit


class Sizer(BaseLego):
    """
    Very small subclass – only overrides the `run()` hook.
    """

    # ------------------------------------------------------------------ #
    # Experiment logic                                                   #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Iterate over all *trainable* parameters, print their shapes and
        output the total amount in **millions**.
        """
        # Grab (name, parameter) tuples from the `legommender` model.
        named_parameters = list(self.legommender.named_parameters())
        # Keep only parameters that will receive gradients.
        named_parameters = [(name, p) for name, p in named_parameters if p.requires_grad]

        for name, p in named_parameters:
            pnt(name, p.data.shape)

        # Sum up the individual element counts and convert to millions.
        num_params = sum(p.numel() for _, p in named_parameters) / 1e6
        pnt(f'Number of parameters: {num_params:.2f}M')


# ---------------------------------------------------------------------- #
# CLI entry point                                                        #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # 1. Parse user supplied arguments                                   #
    # ------------------------------------------------------------------ #
    configuration = CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            exp='config/exp/default.yaml',
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            item_page_size=64,
            batch_size=64,
        ),
    ).parse()

    # ------------------------------------------------------------------ #
    # 2. Run the sizer                                                   #
    # ------------------------------------------------------------------ #
    sizer = Sizer(config=configuration)
    sizer.run()
