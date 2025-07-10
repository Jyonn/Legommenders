"""
monitor.py

A minimalistic *early-stopping* helper.

The class keeps track of the “best” metric value seen so far and decides
whether:

  • the current value is the new best                    -> Symbols.best
  • training should stop because of patience exhausted   -> Symbols.stop
  • nothing special happens (continue training)          -> Symbols.skip

It relies on the boolean flag `minimize` to know if “smaller is better”
(e.g., LogLoss) or “larger is better” (e.g., AUC).
"""

from loader.symbols import Symbols


class Monitor:
    """
    Parameters
    ----------
    minimize : bool
        If True, the goal is to *minimize* the monitored value
        (e.g., loss).  If False, we aim to *maximize* it
        (e.g., accuracy, AUC).
    patience : int, default=2
        How many *consecutive* non-improving steps to tolerate before
        signalling `Symbols.stop`.
    """

    def __init__(self, minimize: bool, patience: int = 2):
        self.patience = patience          # allowed non-improving steps
        self.best_value = None            # best metric value seen so far
        self.minimize = minimize          # optimisation direction
        self.best_index = 0               # step index of best_value
        self.current_index = -1           # step counter

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def push(self, value: float):
        """
        Feed a new metric value and obtain control signal.

        Returns
        -------
        Symbols.best  : new best value achieved
        Symbols.skip  : training continues, no new best
        Symbols.stop  : patience exhausted, time to stop
        """
        self.current_index += 1

        # First value — initialise & mark as best -------------------------
        if self.best_value is None:
            self.best_value = value
            return Symbols.best

        # Check for improvement ------------------------------------------
        # XOR trick:
        #   minimize == True  and value < best_value  -> improvement
        #   minimize == False and value > best_value  -> improvement
        if self.minimize ^ (value > self.best_value):
            self.best_value = value
            self.best_index = self.current_index
            return Symbols.best

        # No improvement – check patience --------------------------------
        if self.current_index - self.best_index >= self.patience:
            return Symbols.stop

        return Symbols.skip
