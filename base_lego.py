import datetime
import multiprocessing
import os
import sys
from typing import Optional

import numpy as np
import pigmento
import torch
from pigmento import pnt
from transformers import get_linear_schedule_with_warmup
from unitok import JsonHandler

from loader.env import Env
from loader.manager import Manager
from loader.symbols import Symbols
from utils import bars
from utils.function import seeding, get_signature
from utils.gpu import GPU
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.path_hub import PathHub


class BaseLego:
    def __init__(self, config):
        self.config = config
        self.data, self.embed, self.model, self.exp = \
            self.config.data, self.config.embed, self.config.model, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm

        config.seed = int(config.seed or 2023)
        seeding(config.seed)

        path_hub = PathHub(
            data_name=self.data.name,
            model_name=self.model.name,
            signature=get_signature(self.data, self.embed, self.model, self.exp)
        )
        Env.set_path_hub(path_hub)

        multiprocessing.set_start_method('fork')
        self.init_pigmento()
        self.prepare_live_experiment()

        JsonHandler.save(self.config(), Env.path_hub.cfg_path)

        pnt('START TIME:', datetime.datetime.now())
        pnt('SIGNATURE:', Env.path_hub.signature)
        pnt('BASE DIR:', Env.path_hub.checkpoint_base_dir)
        pnt('python', ' '.join(sys.argv))

        Env.device = self.get_device()
        Env.simple_dev = self.exp.policy.simple_dev
        # Setting.dataset = self.data.dataset

        self.manager = Manager(
            data=self.data,
            embed=self.embed,
            model=self.model,
            exp=self.exp,
        )

        self.legommender = self.manager.legommender.to(Env.device)
        self.resampler = self.manager.resampler
        self.cacher = self.legommender.cacher

        self.manager.stringify()

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

    def prepare_live_experiment(self):
        pass

    @staticmethod
    def init_pigmento():
        pigmento.add_time_prefix()
        pigmento.add_log_plugin(Env.path_hub.log_path)
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
            use_instance_class='both',
        )

    def init_optimizer(self):
        if self.legommender.config.use_item_content and self.exp.policy.item_lr:
            pnt('split item pretrained encoder parameters')
            pnt('pretrained lr:', self.exp.policy.item_lr)
            pnt('other lr:', self.exp.policy.lr)
            pretrained_parameters, other_parameters = self.legommender.get_parameters()
            self.optimizer = torch.optim.Adam([
                {'params': pretrained_parameters, 'lr': self.exp.policy.item_lr},
                {'params': other_parameters, 'lr': self.exp.policy.lr}
            ])
        else:
            pnt('use single lr:', self.exp.policy.lr)
            self.optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.legommender.parameters()),
                lr=self.exp.policy.lr
            )

            for name, p in self.legommender.named_parameters():  # type: str, torch.Tensor
                if p.requires_grad:
                    pnt(name, p.data.shape)

    def init_scheduler(self):
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            num_training_steps=len(self.manager.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
        )

    def load(self, sign=None):
        if not sign:
            sign = self.exp.load.sign
        if not sign:
            return
        sign = sign.replace('@', '')
        path = os.path.join(Env.path_hub.checkpoint_base_dir, f'{sign}.pt')
        state_dict = torch.load(path, map_location=Env.device)
        # compatible to old version where each operator are wrapped with an encoder
        self.legommender.load_state_dict(state_dict['model'], strict=self.exp.load.strict)
        if not self.exp.load.model_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
        pnt(f"load model from {path}")

    def save(self):
        state_dict = dict(
            model=self.legommender.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
        )
        torch.save(state_dict, Env.path_hub.ckpt_path)
        pnt(f'save model to {Env.path_hub.ckpt_path}')

    def get_device(self):
        cuda = self.config.cuda
        if cuda in ['-1', -1] or cuda is False:
            pnt('choose cpu')
            return 'cpu'
        if isinstance(cuda, int) or isinstance(cuda, str):
            pnt(f'User select cuda {cuda}')
            # return f"cuda:{cuda}"
            cuda = eval(f'[{cuda}]') if isinstance(cuda, str) else cuda
            return torch.cuda.device(cuda)
        return GPU.auto_choose(torch_format=True)

    def log_interval(self, epoch, step, loss):
        # pnt(f'[epoch {epoch}] step {step}, loss {loss:.4f}')
        # with open(self.log_file, 'a+') as f:
        #     f.write(f'{prefix_s} {text}\n')
        with open(Env.path_hub.log_path, 'a+') as f:
            f.write(f'[epoch {epoch}] step {step}, loss {loss:.4f}\n')

    def log_epoch(self, epoch, results):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        pnt(f'[epoch {epoch}] {line}')

    def train_get_user_embedding(self):
        self.manager.get_train_loader(Symbols.test)
        assert self.cacher.user.cached, 'fast eval not enabled'
        user_embeddings = self.cacher.user.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'user_embeddings.npy')
        pnt(f'store user embeddings to {store_path}')
        np.save(store_path, user_embeddings)

    def train_get_item_embedding(self):
        self.cacher.item.cache(self.resampler.item_cache)
        item_embeddings = self.cacher.item.repr.detach().cpu().numpy()
        store_path = os.path.join(self.exp.dir, 'item_embeddings.npy')
        pnt(f'store item embeddings to {store_path}')
        np.save(store_path, item_embeddings)

    def base_evaluate(self, loader, cols, bar: bars.Bar):
        score_series = torch.zeros(len(loader.dataset), dtype=torch.float32)
        col_series = {col: torch.zeros(len(loader.dataset), dtype=torch.long) for col in cols}

        meaner = Meaner()

        index = 0
        for step, batch in enumerate(bar := bar(loader, disable=self.disable_tqdm)):
            Env.latency_timer.run(Symbols.test)
            with torch.no_grad():
                scores = self.legommender(batch=batch)
                if scores.dim() == 2:
                    scores = scores.squeeze(1)
            Env.latency_timer.run(Symbols.test)

            batch_size = scores.size(0)
            for col in cols:
                if batch[col].dim() == 2:
                    col_series[col][index:index + batch_size] = batch[col][:, 0]
                else:
                    col_series[col][index:index + batch_size] = batch[col]
            scores = scores.cpu().detach()
            score_series[index:index + batch_size] = scores
            index += batch_size
            bar.set_postfix_str(f'score: {meaner(scores.mean().item()):.4f}')

        return score_series, col_series

    def evaluate(self, loader, metrics, bar: bars.Bar):
        pool = MetricPool.parse(metrics)
        self.legommender.eval()

        label_col, group_col = self.manager.cm.label_col, self.manager.cm.group_col
        score_series, col_series = self.base_evaluate(loader, cols=[label_col, group_col], bar=bar)
        label_series, group_series = col_series[label_col], col_series[group_col]

        results = pool.calculate(score_series, label_series, group_series)
        return results

    def run(self):
        raise NotImplementedError
