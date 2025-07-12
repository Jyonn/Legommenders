"""
trainer.py

Full *train → dev-eval → test* pipeline including
------------------------------------------------
• model initialisation (handled by `BaseLego`)
• optimiser & LR-scheduler setup
• training loop with gradient accumulation
• periodic *dev* set evaluation (+ early stopping)
• final *test* set evaluation
• optional live experiment tracking via a remote «lego-server»

The class hierarchy looks as follows::

    BaseLego
        ▲
        ├── Tester     (adds evaluate / latency benchmark)
        │        ▲
        │        └── Trainer  ←  THIS FILE
        │
        └── Sizer / Splitter / …

Key features
------------
1. *Live* experiment
   If the user passes `--session <SESSION_ID>` the script connects to a
   lego-server instance, verifies that the local signature / seed match
   the remote record and then periodically pushes updates (log file,
   final metrics, …).

2. *Early stopping*
   Implemented via `utils.monitor.Monitor` which tracks a target metric
   (`exp.store.metric`) and stops training once the validation score
   stops improving for `patience` epochs.

3. *Gradient accumulation*
   The inner loop aggregates gradients over
   `exp.policy.accumulate_batch` mini-batches before calling
   `optimizer.step()`.  This is handy when GPU memory is limited.

4. *Checkpointing*
   The best model according to the monitor metric is stored via
   `self.save()` so that the subsequent `test()` phase can reload it.

CLI usage
---------
python trainer.py \
    --data  movielens \
    --model config/model/bert_recommender.yaml \
    --embed config/embed/bert.yaml \
    --exp   config/exp/default.yaml
"""

from __future__ import annotations

import psutil
import torch
from typing import Dict, Any, Optional

from pigmento import pnt

# Project modules
from loader.env import Env
from loader.symbols import Symbols
from tester import Tester
from utils import bars, io
from utils.config_init import CommandInit
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.monitor import Monitor
from utils.server import Server, ExperimentBody


class Trainer(Tester):
    """
    Adds *training* capability on top of the evaluation logic provided
    by `Tester`.
    """

    # Will be populated in `prepare_live_experiment` if `--session` is set.
    server: Optional[Server] = None

    # ------------------------------------------------------------------ #
    # Live experiment integration                                        #
    # ------------------------------------------------------------------ #
    def prepare_live_experiment(self) -> None:  # override BaseLego hook
        """
        Connect to the lego-server and perform a series of sanity checks
        (signature, seed, duplicate process, …).

        Raises
        ------
        ValueError  If any of the validations fails.
        """
        if not self.config.session:
            return  # local run, nothing to do

        # Authenticate + retrieve meta-information
        self.server = Server.auto_auth()
        experiment_info = self.server.get_experiment_info(session=self.config.session)
        experiment = ExperimentBody(experiment_info.body)

        # Ensure local run matches the server record
        if experiment.signature != Env.ph.signature:
            pnt(f"Signature mismatch: {Env.ph.signature} != {experiment.signature}")
            raise ValueError("Signature mismatch")

        if experiment.seed != self.config.seed:
            pnt(f"Seed mismatch: {self.config.seed} != {experiment.seed}")
            raise ValueError("Seed mismatch")

        if experiment.is_completed:
            pnt(f"Experiment {Env.ph.signature} is already completed.")
            raise ValueError("Experiment already completed")

        if experiment.pid is not None and psutil.pid_exists(experiment.pid):
            pnt(f"Experiment {Env.ph.signature} is already running (pid={experiment.pid}).")
            raise ValueError("Experiment already running")

        # All good – register *this* process id at the server
        self.server.register_experiment(session=self.config.session)

    # ------------------------------------------------------------------ #
    # Dev-set evaluation helpers                                         #
    # ------------------------------------------------------------------ #
    def simple_evaluate(self, dev_bar: bars.Bar):
        """
        Compute the *training* loss over the dev loader.
        Used when `Env.simple_dev` is enabled (i.e. no additional metrics
        requested).
        """
        loader = self.manager.get_dev_loader()
        meaner = Meaner()

        for step, batch in enumerate(bar := dev_bar(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.legommender(batch=batch)
            bar.set_postfix_str(f'loss: {meaner(loss.item()):.4f}')

        return dict(loss=meaner.mean), meaner.mean

    def dev(self, dev_bar: bars.Bar):
        """
        Full dev-set evaluation that calculates the target metric defined
        in the experiment config (`exp.store.metric`).
        """
        assert self.exp.store.metric, "A dev metric must be specified (`exp.store.metric`)."
        loader = self.manager.get_dev_loader()

        results = self.evaluate(loader,
                                metrics=[self.exp.store.metric],
                                bar=dev_bar)
        return results, results[self.exp.store.metric]

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """
        Outer epoch loop + gradient accumulation + early stopping.
        """
        monitor = Monitor(
            patience=self.exp.store.patience,
            minimize=Env.simple_dev or MetricPool.is_minimize(self.exp.store.metric)
        )

        dev_func = self.simple_evaluate if Env.simple_dev else self.dev

        train_steps = len(self.manager.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        # Interval at which we log `loss=` lines to the .log file
        check_interval = self.exp.policy.check_interval
        if check_interval and check_interval < 0:
            # Interpret negative value as "n chunks per epoch"
            check_interval = max(train_steps // (-check_interval), 1)

        meaner = Meaner()
        loader = self.manager.get_train_loader()

        # ------------------------------------------------------------------ #
        # Epoch loop                                                         #
        # ------------------------------------------------------------------ #
        self.optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch):
            self.legommender.train()
            self.manager.setup(Symbols.train)

            for step, batch in enumerate(bar := bars.TrainBar(epoch=epoch)(loader,
                                                                           disable=self.disable_tqdm)):
                # Forward + backward
                loss = self.legommender(batch=batch)
                loss.backward()

                bar.set_postfix_str(f'loss: {meaner(loss.item()):.4f}')

                # ----------- gradient accumulation --------------------------- #
                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                # ----------- optional interval logging ---------------------- #
                if check_interval and (step + 1) % check_interval == 0:
                    self.log_interval(epoch, step, meaner.mean)

                # ----------- limit epoch length if requested ---------------- #
                if self.exp.policy.epoch_batch:
                    if self.exp.policy.epoch_batch < 0:  # relative
                        if step > max(train_steps // (-self.exp.policy.epoch_batch), 1):
                            break
                    else:  # absolute
                        if step > self.exp.policy.epoch_batch:
                            break

            # ---------------------------------------------------------------- #
            # End of epoch → dev evaluation + early stopping decision          #
            # ---------------------------------------------------------------- #
            dev_bar = bars.DevBar(epoch=epoch, train_loss=meaner.mean)
            dev_results, monitor_metric = dev_func(dev_bar)
            self.log_epoch(epoch, dev_results)

            action = monitor.push(monitor_metric)
            if action is Symbols.stop:
                pnt('Early stop triggered.')
                break
            elif action is Symbols.best:
                self.save()

        pnt('Training Ended')

    # ------------------------------------------------------------------ #
    # Overwrite `Tester.test()` to also upload results                    #
    # ------------------------------------------------------------------ #
    def test(self):
        results = super().test()
        self.complete_live_experiment(results)

    # ------------------------------------------------------------------ #
    # Helper: purify console log before uploading                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_pured_log() -> str:
        """
        Remove carriage-return artifacts from the console log so that the
        file is nicely readable in a browser.
        """
        log_bin = io.file_load(Env.ph.log_path, binary=True)
        lines = log_bin.split(b'\n')
        cleaned_lines = [
            line[line.rfind(b'\r') + 1:] if b'\r' in line else line
            for line in lines
        ]
        return b'\n'.join(cleaned_lines).decode('utf-8', errors='replace')

    # ------------------------------------------------------------------ #
    # Notify lego-server that the experiment is finished                  #
    # ------------------------------------------------------------------ #
    def complete_live_experiment(self, results: Dict[str, float]) -> None:
        if not self.config.session:
            return  # local run

        log = self.get_pured_log()
        performance = io.json_dumps(results)

        self.server.complete_experiment(
            session=self.config.session,
            log=log,
            performance=performance
        )
        pnt(f"Experiment {Env.ph.signature} completed and uploaded to lego-server.")

    # ------------------------------------------------------------------ #
    # `BaseLego` hook                                                    #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Full workflow:
        1. optimiser / scheduler init
        2. (optional) checkpoint load
        3. training
        4. reload best checkpoint
        5. final test evaluation
        """
        self.init_optimizer()
        self.init_scheduler()
        self.load()                   # might load previous weights
        self.train()                  # dev evaluation + early stop inside
        self.load(Env.ph.signature)   # reload best model
        self.test()                   # test set evaluation


# ---------------------------------------------------------------------- #
# Convenience wrapper for Hydra / Slurm jobs                             #
# ---------------------------------------------------------------------- #
def get_configurations(kwargs: Optional[Dict[str, Any]] = None):
    """
    Parse CLI arguments but allow overriding via a kwargs dict to make
    the function usable from a Jupyter notebook or another Python script.
    """
    return CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            embed='config/embed/null.yaml',
            exp='config/exp/default.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            item_page_size=64,
        ),
    ).parse(kwargs=kwargs)


# ---------------------------------------------------------------------- #
# Stand-alone CLI                                                        #
# ---------------------------------------------------------------------- #
if __name__ == '__main__':
    configuration = get_configurations()
    trainer = Trainer(config=configuration)
    trainer.run()
