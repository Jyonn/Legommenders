"""
base_lego.py

A *framework–level* helper that wires together the different building
blocks needed for a recommendation-style experiment.

Responsibilities
----------------
1. Parse the user provided `config` object and expose the individual
   sub-sections (data / embed / model / exp).
2. Create run-time artefacts such as:

   • Reproducible seeding
   • File/Directory layout via `PathHub`
   • Logging through «pigmento»
   • CUDA / CPU device selection
   • `Manager` instance that owns
       – the data loaders
       – the model (`legommender`)
       – the resampler / cacher helpers

3. Provide utilities that *most* concrete experiments will need:

   • Optimiser / scheduler initialisation
   • Check-point save & load
   • Periodic logging helpers
   • Convenience methods to dump user/item embeddings
   • A generic evaluation routine that reduces boiler-plate

Only the `run()` method is left abstract – subclasses decide what the
actual training loop looks like.

Note
----
The class is intentionally *stateful*: attributes such as
`self.legommender` and `self.optimizer` are created in `__init__` so
that they can be re-used by sub-classes without having to recreate /
re-calculate expensive objects.
"""

from __future__ import annotations

import datetime
import multiprocessing
import os
import sys
from typing import Optional, Tuple, Dict, cast

import numpy as np
import pigmento
import torch
from pigmento import pnt                        # colourful logging
from transformers import get_linear_schedule_with_warmup

from loader.cacher.item_cacher import ItemCacher
# Local project imports
from loader.env import Env
from loader.manager import Manager
from loader.symbols import Symbols
from utils import bars, io
from utils.function import seeding, get_signature
from utils.gpu import GPU
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.path_hub import PathHub


class BaseLego:
    """
    Abstract experiment scaffold.

    Parameters
    ----------
    config : Any
        A (likely Hydra / argparse) namespace that contains the
        sub-configurations: `data`, `embed`, `model`, `exp`, `seed`, ...
    """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, config):
        # ------------- 0. Parse config -------------------------------- #
        self.config = config
        self.data, self.embed, self.model, self.exp = \
            config.data, config.embed, config.model, config.exp

        # Whether or not tqdm progress bars should be suppressed.
        self.disable_tqdm = self.exp.policy.disable_tqdm

        # ------------- 1. Reproducibility ----------------------------- #
        # Fallback to 2023 if user did not specify a seed.
        config.seed = int(config.seed or 2023)
        seeding(config.seed)                   # sets NumPy / PyTorch / random seed

        # ------------- 2. I/O layout ---------------------------------- #
        # The *signature* encodes which data set / model / hyper-params
        # are used and is therefore unique for an experiment run.
        Env.ph = PathHub(
            data_name=self.data.name,
            model_name=self.model.name,
            signature=get_signature(self.data, self.embed, self.model, self.exp)
        )

        # ------------- 3. Logging (pigmento) -------------------------- #
        multiprocessing.set_start_method('fork', force=True)
        self.init_pigmento()                   # configure pigmento
        self.prepare_live_experiment()         # hook for subclasses

        # Persist *the complete* config so we can reproduce the run later.
        io.json_save(self.config(), Env.ph.cfg_path)

        # Print some run meta-information.
        pnt('START TIME:', datetime.datetime.now())
        pnt('SIGNATURE:', Env.ph.signature)
        pnt('BASE DIR:', Env.ph.checkpoint_base_dir)
        pnt('python', ' '.join(sys.argv))

        # ------------- 4. Device placement --------------------------- #
        Env.device = self.get_device()
        Env.simple_dev = self.exp.policy.simple_dev

        # ------------- 5. Manager & Model ---------------------------- #
        # `Manager` bundles all data-related logic as well as the model.
        self.manager = Manager(
            data=self.data,
            embed=self.embed,
            model=self.model,
            exp=self.exp,
        )

        # Short-cuts for convenience
        self.legommender = self.manager.legommender.to(Env.device)
        self.resampler   = self.manager.resampler
        self.cacher      = self.legommender.cacher

        # Print a human-readable summary of the experiment components.
        self.manager.stringify()

        # ------------- 6. Optimiser / Scheduler (lazy init) ---------- #
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #
    def prepare_live_experiment(self) -> None:
        """
        Optional method for subclasses to override if they want to add
        additional live logging (TensorBoard, WandB, …) or set up other
        resources *before* training starts.
        """
        pass

    # ------------------------------------------------------------------ #
    # Logging initialisation                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def init_pigmento() -> None:
        """
        Configure colourised console logging.
        """
        pigmento.add_time_prefix()
        pigmento.add_log_plugin(Env.ph.log_path)
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
            use_instance_class=True,
        )

    # ------------------------------------------------------------------ #
    # Optimiser / Scheduler                                              #
    # ------------------------------------------------------------------ #
    def init_optimizer(self) -> None:
        """
        Instantiate the Adam optimiser.

        If the model uses *pre-trained* item encoders we often want to
        train them with a lower learning rate than the rest of the
        network. This helper therefore supports a 2-group parameter set-up.
        """
        if self.legommender.config.use_item_content and self.exp.policy.item_lr:
            # ---------------------------------------------------------- #
            # Two learning rates: pretrained encoder vs. task specific   #
            # ---------------------------------------------------------- #
            pnt('split item pretrained encoder parameters')
            pnt('pretrained lr:', self.exp.policy.item_lr)
            pnt('other lr:', self.exp.policy.lr)

            pretrained_parameters, other_parameters = self.legommender.get_parameters()
            self.optimizer = torch.optim.Adam(
                [
                    {'params': pretrained_parameters, 'lr': self.exp.policy.item_lr},
                    {'params': other_parameters,   'lr': self.exp.policy.lr},
                ]
            )
        else:
            # Single learning rate for all trainable parameters
            pnt('use single lr:', self.exp.policy.lr)
            self.optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.legommender.parameters()),
                lr=self.exp.policy.lr
            )

            # Print which parameters actually receive gradients
            for name, p in self.legommender.named_parameters():
                if p.requires_grad:
                    pnt(name, p.data.shape)

    def init_scheduler(self) -> None:
        """
        Linear warm-up learning rate schedule.
        """
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            num_training_steps=(
                len(self.manager.train_set)
                // self.exp.policy.batch_size
                * self.exp.policy.epoch
            ),
        )

    # ------------------------------------------------------------------ #
    # Check-pointing                                                     #
    # ------------------------------------------------------------------ #
    def load(self, sign: Optional[str] = None) -> None:
        """
        Restore model / optimiser / scheduler state.

        Parameters
        ----------
        sign : str, optional
            The *signature* string that identifies the checkpoint. If
            `None`, falls back to `self.exp.load.sign`.
        """
        sign = sign or self.exp.load.sign
        if not sign:
            return  # nothing to load

        sign = sign.replace('@', '')
        path = os.path.join(Env.ph.checkpoint_base_dir, f'{sign}.pt')
        state_dict = torch.load(path, map_location=Env.device)

        # Older versions wrapped each operator in an extra encoder layer –
        # therefore we keep `strict` configurable.
        self.legommender.load_state_dict(state_dict['model'],
                                         strict=self.exp.load.strict)

        if not self.exp.load.model_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

        pnt(f"load model from {path}")

    def save(self) -> None:
        """
        Serialise current experiment state to disk.
        """
        state_dict = dict(
            model=self.legommender.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
        )
        torch.save(state_dict, Env.ph.ckpt_path)
        pnt(f'save model to {Env.ph.ckpt_path}')

    # ------------------------------------------------------------------ #
    # Device management                                                  #
    # ------------------------------------------------------------------ #
    def get_device(self) -> torch.device:
        """
        Decide on which device the experiment should run.

        Follows priority:
        1) Explicit user selection via `config.cuda`
        2) Automatic GPU choice (`GPU.auto_choose`)
        3) CPU fallback
        """
        cuda = self.config.cuda

        # ---- force CPU ------------------------------------------------ #
        if cuda in ['-1', -1] or cuda is False:
            pnt('choose cpu')
            return torch.device('cpu')

        # ---- explicit GPU id(s) --------------------------------------- #
        if isinstance(cuda, (int, str)):
            pnt(f'User select cuda {cuda}')
            # Accept comma separated string, e.g. "0,1"
            cuda_idx = eval(f'[{cuda}]') if isinstance(cuda, str) else cuda
            return torch.device(f'cuda:{cuda_idx[0]}' if isinstance(cuda_idx, list) else f'cuda:{cuda_idx}')

        # ---- automatic selection -------------------------------------- #
        return GPU.auto_choose(torch_format=True)

    # ------------------------------------------------------------------ #
    # Logging helpers                                                    #
    # ------------------------------------------------------------------ #
    def log_interval(self, epoch: int, step: int, loss: float) -> None:
        """
        Append a single training step line to the run log file.
        """
        io.file_save(
            Env.ph.log_path,
            f'[epoch {epoch}] step {step}, loss {loss:.4f}\n',
            append=True
        )

    def log_epoch(self, epoch: int, results: Dict[str, float]) -> None:
        """
        Pretty print epoch-level evaluation results.
        """
        line = ', '.join([f'{metric} {results[metric]:.4f}'
                          for metric in results])
        pnt(f'[epoch {epoch}] {line}')

    # ------------------------------------------------------------------ #
    # Embedding export                                                   #
    # ------------------------------------------------------------------ #
    def train_get_user_embedding(self) -> None:
        """
        Dump *training* user embeddings to disk. Useful for offline
        analysis or visualisation (e.g. t-SNE).
        """
        self.manager.get_train_loader(Symbols.test)         # ensure cacher is filled
        assert self.cacher.user.cached, 'fast eval not enabled'

        user_emb = self.cacher.user.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'user_embeddings.npy')
        pnt(f'store user embeddings to {store_path}')
        np.save(store_path, user_emb)

    def train_get_item_embedding(self) -> None:
        """
        Same as `train_get_user_embedding` but for items.
        """
        item_cacher = cast(ItemCacher, self.cacher.item)
        item_cacher.cache(self.resampler.item_cache)
        item_emb = item_cacher.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'item_embeddings.npy')
        pnt(f'store item embeddings to {store_path}')
        np.save(store_path, item_emb)

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #
    def base_evaluate(
        self,
        loader,                        # DataLoader
        cols,                          # list[str]
        bar: bars.Bar                  # progress-bar factory
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run the model in *inference* mode over the given loader and
        collect raw scores + selected columns so that higher-level
        metrics can be computed.

        Returns
        -------
        score_series : torch.FloatTensor
            Flat tensor of shape (N,) with the model outputs.
        col_series : Dict[str, torch.LongTensor]
            A dict that maps the requested `cols` to equally long tensors.
        """
        score_series = torch.zeros(len(loader.dataset), dtype=torch.float32)
        col_series   = {col: torch.zeros(len(loader.dataset), dtype=torch.long) for col in cols}

        meaner = Meaner()      # running mean for progress update
        index  = 0

        for step, batch in enumerate(bar := bar(loader, disable=self.disable_tqdm)):
            # Measure forward latency if desired
            Env.latency_timer.run(Symbols.test)
            with torch.no_grad():
                scores = self.legommender(batch=batch)
                # Some models output (B,1) – flatten to (B,)
                if scores.dim() == 2:
                    scores = scores.squeeze(1)
            Env.latency_timer.run(Symbols.test)

            batch_size = scores.size(0)

            # Gather additional columns (labels, group ids, …)
            for col in cols:
                if batch[col].dim() == 2:
                    col_series[col][index:index + batch_size] = batch[col][:, 0]
                else:
                    col_series[col][index:index + batch_size] = batch[col]

            scores_cpu = scores.cpu().detach()
            score_series[index:index + batch_size] = scores_cpu
            index += batch_size

            bar.set_postfix_str(f'score: {meaner(scores_cpu.mean().item()):.4f}')

        return score_series, col_series

    def evaluate(
        self,
        loader,                        # DataLoader
        metrics,                       # list[str] | str
        bar: bars.Bar
    ) -> Dict[str, float]:
        """
        Compute *high-level* evaluation metrics (AUC, NDCG, …).

        Steps
        -----
        1. Collect raw model scores + label / group columns via
           `base_evaluate`.
        2. Feed the series into `MetricPool` which performs the actual
           calculation.
        """
        pool = MetricPool.parse(metrics)
        self.legommender.eval()

        label_col, group_col = self.manager.cm.label_col, self.manager.cm.group_col
        score_series, col_series = self.base_evaluate(loader,
                                                      cols=[label_col, group_col],
                                                      bar=bar)
        label_series = col_series[label_col]
        group_series = col_series[group_col]

        results = pool.calculate(score_series, label_series, group_series)
        return results

    # ------------------------------------------------------------------ #
    # Entry-point (to be implemented)                                    #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        *Abstract* – concrete experiment class must implement the
        training / evaluation loop here.
        """
        raise NotImplementedError
