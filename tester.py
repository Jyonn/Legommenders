"""
tester.py

End-to-end evaluation / latency benchmark
=========================================

This script is a thin orchestration layer that *reuses* the heavy
initialisation performed by `BaseLego` (data-loading, model creation,
device selection, …) and adds three small utilities:

1. `test()`   Run the model on the test split and compute the metrics
              specified in the experiment config (`exp.metrics`).

2. `latency()`  
   Benchmark the model’s **inference** time by recording how long each
   forward pass takes.  
   The feature can be enabled via the CLI flag `--latency`.

3. `run()`    The public entry-point:  
   • restores a checkpoint (if `--exp.load.*` is set)  
   • dispatches to `test()` *or* `latency()` depending on the CLI flags.


Why not place the logic directly in `BaseLego`?
----------------------------------------------
We keep `BaseLego` completely framework-agnostic.  
Specialized tasks such as *only* testing or *sizing* the model are
implemented in a dedicated subclass so that each script has a single,
clearly defined purpose.
"""

from __future__ import annotations

from typing import Dict

from pigmento import pnt

from base_lego import BaseLego
from loader.env import Env
from loader.symbols import Symbols
from utils import bars, io
from utils.config_init import CommandInit
from utils.timer import StatusTimer


class Tester(BaseLego):
    """
    Sub-class that adds evaluation / latency benchmarking capability.
    """

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #
    def test(self) -> Dict[str, float]:
        """
        Run the model on the *test* split and compute all requested
        metrics.

        The results are printed to the console **and** persisted to
        `Env.ph.result_path` so they can be picked up by downstream
        scripts or a hyper-parameter sweep manager.
        """
        loader = self.manager.get_test_loader()
        results = self.evaluate(
            loader,
            metrics=self.exp.metrics,
            bar=bars.TestBar()
        )

        # Pretty print & save to file
        lines = []
        for metric, value in results.items():
            pnt(f'{metric}: {value:.4f}')
            lines.append(f'{metric}: {value:.4f}')

        io.file_save(Env.ph.result_path, '\n'.join(lines))
        return results

    # ------------------------------------------------------------------ #
    # Latency benchmark                                                  #
    # ------------------------------------------------------------------ #
    def latency(self) -> None:
        """
        Measure *pure forward pass* latency.

        The function runs `self.test()` while keeping `Env.latency_timer`
        active. Afterwards it prints the average milliseconds per
        mini-batch.
        """
        # Prepare timer
        Env.latency_timer.activate()
        Env.latency_timer.clear()
        Env.latency_timer.set_total_count(
            Symbols.test,
            self.config.num_batches or 0
        )

        try:
            self.test()
        except (KeyboardInterrupt, StopIteration):
            # Allow graceful abort during long benchmarks
            pass
        finally:
            st: StatusTimer = Env.latency_timer.status_dict[Symbols.test]
            pnt(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

    # ------------------------------------------------------------------ #
    # `BaseLego` hook                                                    #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Restore checkpoint (if requested) and start either
        • the normal *evaluation* mode or
        • the *latency* benchmark mode.
        """
        self.load()   # may be a no-op if `exp.load.sign` is empty

        if self.config.latency:
            self.latency()
        else:
            self.test()


# ---------------------------------------------------------------------- #
# CLI entry-point                                                        #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            exp='config/exp/default.yaml',
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            latency=False,     # turn on latency benchmark
            num_batches=1000,  # only relevant for latency mode
        ),
    ).parse()

    tester = Tester(config=configuration)
    tester.run()
