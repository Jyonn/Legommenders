import datetime
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import pigmento
import torch
from pigmento import pnt
from oba import Obj
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.meta import Meta, Phases
from loader.controller import Controller
# from loader.mode_hub import ModeHub
from model.operators.base_llm_operator import BaseLLMOperator
from utils.config_init import ConfigInit
from utils.function import seeding
from utils.gpu import GPU
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.monitor import Monitor
from loader.pager.llm_split_pager import LLMSplitPager
from utils.structure import Structure
from utils.submission import Submission

torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Worker:
    def __init__(self, config):
        self.config = config
        self.data, self.embed, self.model, self.exp = \
            self.config.data, self.config.embed, self.config.model, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm
        self.mode = self.exp.mode.lower()

        config.seed = int(config.seed or 2023)
        seeding(config.seed)

        self.init_pigmento()

        pnt('START TIME:', datetime.datetime.now())
        # print command line arguments
        pnt('python ', ' '.join(sys.argv))
        pnt(json.dumps(Obj.raw(self.config), indent=4))

        Meta.device = self.get_device()
        Meta.simple_dev = self.exp.policy.simple_dev
        # Setting.dataset = self.data.dataset

        self.controller = Controller(
            data=self.data,
            embed=self.embed,
            model=self.model,
            exp=self.exp,
        )

        self.legommender = self.controller.legommender.to(Meta.device)
        self.resampler = self.controller.resampler
        self.cacher = self.legommender.cacher
        self.cacher.activate(config.fast_eval)
        self.load_path = self.parse_load_path()

        pnt(self.controller.depots.a_depot()[0])
        pnt(Structure().analyse_and_stringify(self.controller.sets.a_set()[0]))

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

        # self.mode_hub = ModeHub(controller=self.controller)

    def init_pigmento(self):
        pigmento.add_time_prefix()
        pigmento.add_log_plugin(self.exp.log)
        pigmento.add_dynamic_color_plugin()
        pnt.set_display_mode(
            display_method_name=False,
            display_class_name=True,
        )

    def load(self, path):
        while True:
            pnt(f"load model from exp {path}")
            try:
                state_dict = torch.load(path, map_location=Meta.device)
                break
            except Exception as e:
                if not self.exp.load.wait_load:
                    raise e
                time.sleep(60)

        # compatible to old version where each operator are wrapped with an encoder
        model_ckpt = dict()
        for key, value in state_dict['model'].items():
            model_ckpt[key.replace('operator.', '')] = value

        self.legommender.load_state_dict(model_ckpt, strict=self.exp.load.strict)
        if not self.exp.load.model_only:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def parse_load_path(self):
        if not self.exp.load.save_dir:
            return

        save_dir = os.path.join(self.exp.dir, self.exp.load.save_dir)
        epochs = Obj.raw(self.exp.load.epochs)
        if not epochs:
            epochs = json.load(open(os.path.join(save_dir, 'candidates.json')))
        elif isinstance(epochs, str):
            epochs = eval(epochs)
        assert isinstance(epochs, list), ValueError(f'fail loading epochs: {epochs}')

        return [os.path.join(save_dir, f'epoch_{epoch}.bin') for epoch in epochs]

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

    @staticmethod
    def log_interval(epoch, step, loss):
        pnt(f'[epoch {epoch}] step {step}, loss {loss:.4f}')

    @staticmethod
    def log_epoch(epoch, results):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        pnt(f'[epoch {epoch}] {line}')

    def train(self) -> int:
        monitor_kwargs = Obj.raw(self.exp.store)

        dev_func = self.dev
        if Meta.simple_dev:
            monitor_kwargs['maximize'] = False
            dev_func = self.simple_evaluate
            pnt('activate simple dev mode')

        monitor = Monitor(
            save_dir=self.exp.dir,
            **monitor_kwargs,
        )

        train_steps = len(self.controller.sets.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        loader = self.controller.get_loader(Phases.train).train()
        self.optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.legommender.train()
            loader.train()
            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                # if step >= 1000:
                #     break
                loss = self.legommender(batch=batch)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                if self.exp.policy.check_interval:
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, loss.item())
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, loss.item())

                if self.exp.policy.epoch_batch:
                    if self.exp.policy.epoch_batch < 0:  # step part
                        if step > max(train_steps // (-self.exp.policy.epoch_batch), 1):
                            break
                    else:
                        if step > self.exp.policy.epoch_batch:
                            break

            dev_results, monitor_metric = dev_func()
            self.log_epoch(epoch, dev_results)

            state_dict = dict(
                model=self.legommender.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
            )
            early_stop = monitor.push(
                epoch=epoch,
                metric=monitor_metric,
                state_dict=state_dict,
            )
            if early_stop == -1:
                return monitor.get_best_epoch()

        pnt('Training Ended')
        monitor.export()

        return monitor.get_best_epoch()

    def dev(self):
        assert self.exp.store.metric
        loader = self.controller.get_loader(Phases.dev).eval()

        results = self.evaluate(loader, metrics=[self.exp.store.metric])
        return results, results[self.exp.store.metric]

    def test_fake(self):
        self.legommender.eval()
        loader = self.controller.get_loader(Phases.test).test()
        loader.dataset.timer.clear()

        score_series, label_series, group_series, fake_series = [], [], [], []
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                scores = self.legommender(batch=batch)
            labels = batch[self.controller.column_map.label_col].tolist()
            groups = batch[self.controller.column_map.group_col].tolist()
            fakes = batch[self.controller.column_map.fake_col].tolist()
            score_series.extend(scores.cpu().detach().tolist())
            label_series.extend(labels)
            group_series.extend(groups)
            fake_series.extend(fakes)

        df = pd.DataFrame(dict(
            score=score_series,
            label=label_series,
            group=group_series,
            fake=fake_series,
        ))
        groups = df.groupby('fake')
        pnt('group by fake done')

        for fake, group in groups:
            pnt('inactive users' if fake else 'active users')
            pool = MetricPool.parse(self.exp.metrics)
            results = pool.calculate(group['score'].tolist(), group['label'].tolist(), group['group'].tolist())
            for metric in results:
                pnt(f'{metric}: {results[metric]:.4f}')

    def mind_large_evaluate(self, loader):
        self.legommender.eval()

        # group_series = submission.depot.data[self.config_manager.column_map.group_col].tolist()
        # item_series = submission.depot.data[self.config_manager.column_map.candidate_col].tolist()
        # score_series = [random.random() for _ in range(len(submission.depot))]
        item_col, group_col = self.controller.column_map.candidate_col, self.controller.column_map.group_col
        score_series, col_series = self.base_evaluate(loader, cols=[item_col, group_col])
        item_series, group_series = col_series[item_col], col_series[group_col]
        # item_series = [v[0] for v in item_series]

        loader.dataset.timer.summarize()

        submission = Submission(
            depot=self.controller.depots[Phases.test],
            column_map=self.controller.column_map,
        )

        export_dir = submission.run(
            scores=score_series,
            groups=group_series,
            items=item_series,
            model_name=self.model.name,
        )

        pnt(f'export to {export_dir}')

    def test(self):
        loader = self.controller.get_loader(Phases.test).test()

        if self.config.mind_large_submission:
            return self.mind_large_evaluate(loader)

        results = self.evaluate(loader, metrics=self.exp.metrics)
        for metric in results:
            pnt(f'{metric}: {results[metric]:.4f}')

    def train_get_user_embedding(self):
        self.controller.get_loader(Phases.train).test()
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

    def base_evaluate(self, loader, cols):
        score_series = torch.zeros(len(loader.dataset), dtype=torch.float32)
        col_series = {col: torch.zeros(len(loader.dataset), dtype=torch.long) for col in cols}

        index = 0
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                scores = self.legommender(batch=batch)
                if scores.dim() == 2:
                    scores = scores.squeeze(1)

            batch_size = scores.size(0)
            for col in cols:
                if batch[col].dim() == 2:
                    col_series[col][index:index + batch_size] = batch[col][:, 0]
                else:
                    col_series[col][index:index + batch_size] = batch[col]
            score_series[index:index + batch_size] = scores.cpu().detach()
            index += batch_size

        return score_series, col_series

    def evaluate(self, loader, metrics):
        pool = MetricPool.parse(metrics)
        self.legommender.eval()

        label_col, group_col = self.controller.column_map.label_col, self.controller.column_map.group_col
        score_series, col_series = self.base_evaluate(loader, cols=[label_col, group_col])
        label_series, group_series = col_series[label_col], col_series[group_col]

        results = pool.calculate(score_series, label_series, group_series)
        return results

    def simple_evaluate(self):
        loader = self.controller.get_loader(Phases.dev).eval()
        total_loss = Meaner()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.legommender(batch=batch)
            total_loss.add(loss.item())

        total_loss = total_loss.mean()
        return dict(loss=total_loss), total_loss

    def train_runner(self):
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

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            num_training_steps=len(self.controller.sets.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
        )

        if self.load_path:
            self.load(self.load_path[0])
        return self.train()

    def test_runner(self):
        self.test()

    def iter_runner(self, handler):
        if self.load_path:
            for path in self.load_path:
                self.load(path)
                handler()
        else:
            handler()

    def test_size(self):
        named_parameters = list(self.legommender.named_parameters())
        # filter out the parameters that don't require a gradient
        named_parameters = [(name, p) for (name, p) in named_parameters if p.requires_grad]
        # list of (name, parameter) pairs
        for (name, p) in named_parameters:
            pnt(name, p.data.shape)
        num_params = sum([p.numel() for (_, p) in named_parameters])
        # to a million
        num_params /= 1e6
        pnt(f'Number of parameters: {num_params:.2f}M')

    def test_llm_layer_split(self):
        item_encoder = self.legommender.item_encoder  # type: BaseLLMOperator
        assert isinstance(item_encoder, BaseLLMOperator), 'llama operator not found'

        pager = LLMSplitPager(
            inputer=item_encoder.inputer,
            layers=Obj.raw(self.exp.store.layers),
            hidden_size=item_encoder.config.embed_hidden_size,
            contents=self.resampler.item_cache,
            model=item_encoder.get_all_hidden_states,
            page_size=self.config.page_size,
        )

        pager.run()
        pager.store(self.exp.store.dir)

    def run(self):
        # params = Obj.raw(self.exp.params)
        # if not isinstance(params, dict):
        #     params = dict()
        #
        # mode_worker = self.mode_hub(self.mode)
        # if mode_worker.load_model:

        if self.mode == 'train':
            self.train_runner()
        elif self.exp.mode == 'test':
            self.iter_runner(self.test_runner)
            # self.test()
        elif self.exp.mode == 'test_fake':
            self.iter_runner(self.test_fake)
        elif self.mode == 'train_test':
            epoch = self.train_runner()
            self.load(os.path.join(self.exp.dir, f'epoch_{epoch}.bin'))
            self.test_runner()
        elif self.mode == 'test_size':
            self.test_size()
        elif self.mode == 'test_llm_layer_split':
            self.test_llm_layer_split()
        elif self.mode == 'train_get_user_embedding':
            if self.load_path:
                self.load(self.load_path[0])
            self.train_get_user_embedding()
        elif self.mode == 'train_get_item_embedding':
            self.load(self.load_path[0])
            self.train_get_item_embedding()

        # self.mode_hub(self.mode)()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model', 'exp'],
        default_args=dict(
            embed='config/embed/null.yaml',
            warmup=0,
            fast_eval=True,
            simple_dev=False,
            batch_size=64,
            acc_batch=1,
            lora=1,
            lora_r=32,
            lr=0.0001,
            item_lr=0.00001,
            mind_large_submission=False,
            hidden_size=64,
            epoch_batch=0,
            max_item_batch_size=0,
            page_size=512,
            patience=2,
            epoch_start=0,
            frozen=True,
            load_path=None,
        ),
        makedirs=[
            'exp.dir',
        ]
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
