"""
trainer.py

Full *train → dev-eval → test* pipeline including
------------------------------------------------
• model initialisation (handled by `BaseLego`)
• optimizer & LR-scheduler setup
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

from contextlib import nullcontext
from itertools import islice

import psutil
import torch
from typing import Dict, Any, Optional

from pigmento import pnt
from transformers import get_linear_schedule_with_warmup

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

    def __init__(self, config):
        policy = config.exp.policy
        fast = config().get('fast')
        if fast is None:
            fast = policy().get('fast')
        if fast is not None and not isinstance(fast, bool):
            raise ValueError('fast must be true or false')
        # Record opt-in precision before BaseLego computes the run signature.
        if fast or 'fast' in policy:
            policy.fast = bool(fast)
        super().__init__(config)

    def get_device(self) -> torch.device:
        device = super().get_device()
        if not getattr(self.exp.policy, 'fast', False):
            return device
        device = torch.device(device)
        if device.type == 'cuda' and torch.cuda.is_available():
            with torch.cuda.device(device):
                if torch.cuda.is_bf16_supported(including_emulation=False):
                    return device
        raise ValueError('--fast true requires a CUDA GPU with native BF16 support')

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
    def _batches_per_epoch(self, loader) -> int:
        count = len(loader)
        limit = getattr(self.exp.policy, 'epoch_batch', None) or 0
        if limit < 0:
            return max(1, count // abs(limit)) if count else 0
        return min(count, limit) if limit else count

    def _accumulation_batches(self) -> int:
        count = getattr(self.exp.policy, 'accumulate_batch', None) or 1
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('accumulate_batch must be a positive integer')
        return count

    def init_scheduler(self) -> None:
        batches = self._batches_per_epoch(self.manager.get_train_loader())
        if not batches:
            raise ValueError('Training requires at least one batch per epoch')
        accumulate = self._accumulation_batches()
        updates_per_epoch = (batches + accumulate - 1) // accumulate
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            num_training_steps=updates_per_epoch * self.exp.policy.epoch,
        )

    def train(self) -> None:
        """
        Outer epoch loop + gradient accumulation + early stopping.
        """
        monitor = Monitor(
            patience=self.exp.store.patience,
            minimize=Env.simple_dev or MetricPool.is_minimize(self.exp.store.metric)
        )

        dev_func = self.simple_evaluate if Env.simple_dev else self.dev

        loader = self.manager.get_train_loader()
        train_steps = self._batches_per_epoch(loader)
        if not train_steps:
            raise ValueError('Training requires at least one batch per epoch')
        accumulate_batch = self._accumulation_batches()
        label_col = self.manager.cm.label_col
        fast = getattr(self.exp.policy, 'fast', False)
        if fast:
            pnt('Fast mode: BF16 mixed precision for training')

        # Interval at which we log `loss=` lines to the .log file
        check_interval = self.exp.policy.check_interval
        if check_interval and check_interval < 0:
            # Interpret negative value as "n chunks per epoch"
            check_interval = max(train_steps // (-check_interval), 1)

        meaner = Meaner()

        # ------------------------------------------------------------------ #
        # Epoch loop                                                         #
        # ------------------------------------------------------------------ #
        self.optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch):
            self.manager.setup(Symbols.train)
            samples_in_update = 0
            bar = bars.TrainBar(epoch=epoch)(
                islice(loader, train_steps), total=train_steps, disable=self.disable_tqdm,
            )
            for step, batch in enumerate(bar):
                with torch.autocast('cuda', dtype=torch.bfloat16) if fast else nullcontext():
                    loss = self.legommender(batch=batch)
                if accumulate_batch == 1:
                    loss.backward()
                else:
                    # The model returns the mean loss of each microbatch.
                    batch_samples = batch[label_col].size(0)
                    (loss * batch_samples).backward()
                    samples_in_update += batch_samples
                bar.set_postfix_str(f'loss: {meaner(loss.item()):.4f}')

                batch_number = step + 1
                if batch_number % accumulate_batch == 0 or batch_number == train_steps:
                    if accumulate_batch > 1:
                        for group in self.optimizer.param_groups:
                            for param in group['params']:
                                if param.grad is not None:
                                    param.grad.div_(samples_in_update)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    samples_in_update = 0

                # ----------- optional interval logging ---------------------- #
                if check_interval and (step + 1) % check_interval == 0:
                    self.log_interval(epoch, step, meaner.mean)

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
