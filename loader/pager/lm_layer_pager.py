"""
lm_layer_pager.py

Sub-class of `BasePager` that extracts and caches *intermediate* hidden
states of a language model (LM) for every element in `contents`.

Motivation
----------
When the same LM has to be queried multiple times (e.g. during
hyper-parameter search or for several downstream heads) it is wasteful
to perform the complete forward pass over and over again.  In many
scenarios it is sufficient to compute the LM once, cache the hidden
states of the layers of interest, and re-use them later on.  
`LMLayerPager` automates exactly this workflow.

Typical workflow
----------------
1) A list of *items* (e.g. user utterances) is handed over via
   `contents` when the pager is instantiated (handled by `BasePager`).

2) For every mini-batch the pager
       • creates input tensors via the provided `ConcatInputer`
       • forwards the batch through the supplied language model  
         (the model itself is passed in the `BasePager` kwargs)
       • extracts the hidden states of the layers listed in `self.layers`
       • stores them in the pre-allocated tensor `self.final_features`
         with shape (num_layers, num_items, seq_len, hidden_size)
       • caches the attention masks in `self.final_masks`

The pager keeps *all* cached tensors on CPU to avoid GPU memory
pressure; make sure to `.to(device)` when reading the cache.

"""

from __future__ import annotations

import torch

from model.inputer.concat_inputer import ConcatInputer
from loader.pager.base_pager import BasePager


class LMLayerPager(BasePager):
    """
    Pager that populates

        • `self.final_features` : hidden states of requested layers
        • `self.final_masks`    : matching attention masks

    batch-by-batch.

    Parameters
    ----------
    inputer : ConcatInputer
        Helper that maps a *single* content element to
            embeddings     : Tensor(seq_len, d_model)
            attention_mask : Tensor(seq_len)
    layers : list[int]
        Indices of the LM layers whose hidden states should be cached.
    hidden_size : int
        Dimensionality of the LM hidden states.
    **kwargs
        Forwarded verbatim to `BasePager`, must include at least
            contents  : list
            model     : Callable (LM that returns hidden_states=list[Tensor])
            page_size : int
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        inputer: ConcatInputer,
        layers: list[int],
        hidden_size: int,
        **kwargs,
    ):
        # Let BasePager set up slicing / progress bar
        super().__init__(desc="Language Model Layer Caching", **kwargs)

        self.inputer = inputer
        self.layers = layers

        # ------------------------------------------------------------------
        # Pre-allocate the destination buffers on *CPU* to keep GPU memory
        # usage low (the `.run()` loop or downstream code can move them
        # back to GPU if required).
        # ------------------------------------------------------------------
        self.final_features = torch.zeros(
            len(layers),                  # layer dimension
            len(self.contents),           # item dimension
            self.inputer.max_sequence_len,
            hidden_size,
            dtype=torch.float,
        )
        self.final_masks = torch.zeros(
            len(self.contents),
            self.inputer.max_sequence_len,
            dtype=torch.long,
        )

    # ------------------------------------------------------------------ #
    # BasePager overrides                                                #
    # ------------------------------------------------------------------ #
    def get_features(self, content, index: int) -> dict:
        """
        Convert a single *content* element into the LM input tensors.

        Returns
        -------
        dict
            inputs_embeds   : Tensor(seq_len, d_model)
            attention_mask  : Tensor(seq_len)
        Both tensors are
            • detached from any computation graph
            • moved to CPU (→ will be copied to GPU inside the model)
        """
        return dict(
            inputs_embeds=self.inputer.get_embeddings(content)
                          .cpu()
                          .detach(),
            attention_mask=self.inputer.get_mask(content)
                            .cpu()
                            .detach(),
        )

    def combine(
        self,
        slices: slice,
        features: dict[str, torch.Tensor],
        output: list[torch.Tensor],
    ) -> None:
        """
        Persist the LM outputs of the current batch.

        Parameters
        ----------
        slices : slice
            *Global* slice covering the indices of `contents`
            that belong to the current batch.
        features : dict
            Batch input as returned by `get_features`
            (only needed here for the attention_mask).
        output : list[Tensor]
            `hidden_states` produced by the language model.
            Shape of one entry: (batch, seq_len, hidden_size)
        """
        # Store requested layer outputs
        for idx, layer_idx in enumerate(self.layers):
            # Note: move to CPU + detach to avoid holding onto the graph
            self.final_features[idx][slices] = output[layer_idx].cpu().detach()

        # Store matching attention masks (already on CPU)
        self.final_masks[slices] = features["attention_mask"].detach()
