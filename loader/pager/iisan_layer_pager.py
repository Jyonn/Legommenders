"""
iisan_layer_pager.py

Sub-class of `BasePager` that caches *layer–wise sentence states* needed
by IISAN (Iterative Interaction & Self-Attention Network).  
While `LMLayerPager` keeps the full sequence of hidden states for each
layer, `IISANLayerPager` only stores one **sentence-level vector per
layer** (shape: num_layers × hidden_size) which is what the IISAN
encoder expects as input.

Typical workflow
----------------
1) A list of sentences (or any token sequences) is handed over via
   `contents` when the pager is instantiated (handled by `BasePager`).

2) Per mini-batch the pager
       • converts the items into embeddings / masks via `ConcatInputer`
       • forwards the batch through the provided IISAN model
         (must return a tensor with shape
          (batch, num_layers, hidden_size))
       • writes the result into `self.final_states`

All cached tensors live on *CPU* to avoid GPU memory pressure.

"""

from __future__ import annotations

import torch

from model.inputer.concat_inputer import ConcatInputer
from loader.pager.base_pager import BasePager


class IISANLayerPager(BasePager):
    """
    Pager that populates `self.final_states`
        shape: (num_items, num_layers, hidden_size).

    Parameters
    ----------
    inputer : ConcatInputer
        Turns a single `content` element into
            • inputs_embeds   : Tensor(seq_len, d_model)
            • attention_mask  : Tensor(seq_len)
    num_layers : int
        Number of IISAN layers whose sentence states should be cached.
    hidden_size : int
        Dimensionality of each sentence state vector.
    **kwargs
        Forwarded to `BasePager`. Expected keys:
            contents  : list
            model     : Callable  (IISAN encoder)
            page_size : int
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        inputer: ConcatInputer,
        num_layers: int,
        hidden_size: int,
        **kwargs,
    ):
        # Provide a nice progress-bar description to BasePager
        super().__init__(desc="Language Model Layer Caching for IISAN", **kwargs)

        self.inputer = inputer
        self.num_layers = num_layers

        # Pre-allocate the destination tensor on *CPU*.
        # layout: (item, layer, hidden)
        self.final_states = torch.zeros(
            len(self.contents),
            num_layers,
            hidden_size,
            dtype=torch.float,
        )

    # ------------------------------------------------------------------ #
    # BasePager overrides                                                #
    # ------------------------------------------------------------------ #
    def get_features(self, content, index: int) -> dict:
        """
        Convert one `content` element into IISAN input tensors.

        Returns
        -------
        dict
            inputs_embeds   : Tensor(seq_len, d_model)
            attention_mask  : Tensor(seq_len)
        Both tensors are detached and moved to CPU immediately.
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
        output: torch.Tensor,
    ) -> None:
        """
        Persist the IISAN outputs of the current batch.

        Parameters
        ----------
        slices : slice
            *Global* indices of `contents` that belong to the current
            batch. Used for direct assignment into the cache.
        features : dict
            Not needed here aside from complying with the BasePager
            signature (attention masks would be available if required).
        output : Tensor
            IISAN sentence states with shape
            (batch, num_layers, hidden_size)
        """
        # Detach to avoid holding the computation graph and move to CPU.
        self.final_states[slices] = output.cpu().detach()
