import datetime
import json
import os
import time

import pandas as pd
import torch
from oba import Obj
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.global_setting import Setting
from loader.config_manager import ConfigManager, Phases
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logger import Logger
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.monitor import Monitor
from utils.printer import printer, Color, Printer
from utils.random_seed import seeding
from utils.structure import Structure


torch.autograd.set_detect_anomaly(True)


class Worker:
    def __init__(self, config):
        self.config = config
        self.data, self.embed, self.model, self.exp = \
            self.config.data, self.config.embed, self.config.model, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm
        self.mode = self.exp.mode.lower()

        config.seed = int(config.seed or 2023)
        seeding(config.seed)

        self.print = printer[('MAIN', '·', Color.CYAN)]
        Printer.logger = Logger(self.exp.log)
        self.print('START TIME:', datetime.datetime.now())
        self.print(json.dumps(Obj.raw(self.config), indent=4))

        Setting.device = self.get_device()

        self.config_manager = ConfigManager(
            data=self.data,
            embed=self.embed,
            model=self.model,
            exp=self.exp,
        )

        self.recommender = self.config_manager.recommender.to(Setting.device)
        self.manager = self.config_manager.manager
        self.load_path = self.parse_load_path()

        Setting.status = self.manager.status

        self.print(self.config_manager.depots.a_depot()[0])
        self.print(Structure().analyse_and_stringify(self.config_manager.sets.a_set()[0]))

    def load(self, path):
        while True:
            self.print(f"load model from exp {path}")
            try:
                state_dict = torch.load(path, map_location=Setting.device)
                break
            except Exception as e:
                if not self.exp.load.wait_load:
                    raise e
                time.sleep(60)

        # compatible to old version where each operator are wrapped with an encoder
        model_ckpt = dict()
        for key, value in state_dict['model'].items():
            model_ckpt[key.replace('operator.', '')] = value

        self.recommender.load_state_dict(model_ckpt, strict=self.exp.load.strict)
        if not self.exp.load.model_only:
            self.m_optimizer.load_state_dict(state_dict['optimizer'])
            self.m_scheduler.load_state_dict(state_dict['scheduler'])

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
        if cuda in ['-1', -1, False]:
            self.print('choose cpu')
            return 'cpu'
        if not cuda:
            return GPU.auto_choose(torch_format=True)
        return f"cuda:{cuda}"

    def log_interval(self, epoch, step, loss):
        self.print[f'epoch {epoch}'](f'step {step}, loss {loss:.4f}')

    def log_epoch(self, epoch, loss):
        self.print[f'epoch {epoch}'](f'loss {loss:.4f}')

    def train(self):
        monitor = Monitor(
            save_dir=self.exp.dir,
            **Obj.raw(self.exp.store)
        )

        train_steps = len(self.config_manager.sets.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        loader = self.config_manager.get_loader(Phases.train).train()
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.recommender.train()
            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
                # if step >= 1000:
                #     break
                loss = self.recommender(batch=batch)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.m_optimizer.step()
                    self.m_scheduler.step()
                    self.m_optimizer.zero_grad()
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
            dev_loss = self.dev()
            self.log_epoch(epoch, dev_loss)

            state_dict = dict(
                model=self.recommender.state_dict(),
                optimizer=self.m_optimizer.state_dict(),
                scheduler=self.m_scheduler.state_dict(),
            )
            early_stop = monitor.push(
                epoch=epoch,
                loss=dev_loss,
                state_dict=state_dict,
            )
            if early_stop == -1:
                return

        self.print('Training Ended')
        monitor.export()

    def dev(self, steps=None):
        self.recommender.eval()
        loader = self.config_manager.get_loader(Phases.dev).eval()

        meaner = Meaner()
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.recommender(batch=batch)  # [B, neg+1]

            meaner.add(loss.item())

            if steps and step >= steps:
                break

        return meaner.mean()

    def test_fake(self):
        self.recommender.eval()
        loader = self.config_manager.get_loader(Phases.test).test()

        score_series, label_series, group_series, fake_series = [], [], [], []
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                scores = self.recommender(batch=batch).squeeze(1)
            labels = batch[self.config_manager.column_map.label_col].tolist()
            groups = batch[self.config_manager.column_map.group_col].tolist()
            fakes = batch[self.config_manager.column_map.fake_col].tolist()
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
        self.print('group by fake done')

        for fake, group in groups:
            self.print('inactive users' if fake else 'active users')
            pool = MetricPool.parse(self.exp.metrics)
            results = pool.calculate(group['score'].tolist(), group['label'].tolist(), group['group'].tolist())
            for metric in results:
                self.print(f'{metric}: {results[metric]:.4f}')
        #
        # meta_groups = [dict(), dict()]
        # for i, (score, label, group, fake) in enumerate(zip(score_series, label_series, group_series, fake_series)):
        #     meta_groups[fake] = meta_groups[fake] or dict()
        #     meta_groups[fake]['group'] = meta_groups[fake].get('group', []) + [group]
        #     meta_groups[fake]['score'] = meta_groups[fake].get('score', []) + [score]
        #     meta_groups[fake]['label'] = meta_groups[fake].get('label', []) + [label]
        #
        # for fake, meta_group in enumerate(meta_groups):
        #     pool = MetricPool.parse(self.exp.metrics)
        #     results = pool.calculate(meta_group['score'], meta_group['label'], meta_group['group'])
        #     self.print('inactive users' if fake else 'active users')
        #     for metric in results:
        #         self.print(f'{metric}: {results[metric]}')

    def test(self, steps=None):
        pool = MetricPool.parse(self.exp.metrics)

        self.recommender.eval()
        loader = self.config_manager.get_loader(Phases.test).test()

        score_series, label_series, group_series = [], [], []
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                scores = self.recommender(batch=batch).squeeze(1)
            labels = batch[self.config_manager.column_map.label_col].tolist()
            groups = batch[self.config_manager.column_map.group_col].tolist()
            score_series.extend(scores.cpu().detach().tolist())
            label_series.extend(labels)
            group_series.extend(groups)

            if steps and step >= steps:
                break

        results = pool.calculate(score_series, label_series, group_series)
        for metric in results:
            self.print(f'{metric}: {results[metric]}')

    def train_runner(self):
        self.m_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.recommender.parameters()),
            lr=self.exp.policy.lr
        )
        self.m_scheduler = get_linear_schedule_with_warmup(
            self.m_optimizer,
            num_warmup_steps=self.exp.policy.n_warmup,
            num_training_steps=len(self.config_manager.sets.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
        )

        self.print('training params')
        total_memory = 0
        for name, p in self.recommender.named_parameters():  # type: str, torch.Tensor
            total_memory += p.element_size() * p.nelement()
            if p.requires_grad:
                self.print(name, p.data.shape)

        if self.load_path:
            self.load(self.load_path[0])
        self.train()

    def dev_runner(self):
        loss_depot = self.dev(10)
        self.log_epoch(0, loss_depot)

    def test_runner(self):
        self.test()

    def iter_runner(self, handler):
        for path in self.load_path:
            self.load(path)
            handler()

    def run(self):
        if self.mode == 'train':
            self.train_runner()
        elif self.mode == 'dev':
            self.iter_runner(self.dev_runner)
        elif self.exp.mode == 'test':
            self.iter_runner(self.test_runner)
        elif self.exp.mode == 'test_fake':
            self.iter_runner(self.test_fake)
        elif self.mode == 'train_test':
            self.train_runner()
            self.test_runner()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model', 'exp', 'embed'],
        makedirs=[
            'exp.dir',
        ]
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
