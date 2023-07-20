import datetime
import json
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch
from oba import Obj
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.global_setting import Setting
from loader.config_manager import ConfigManager, Phases
from model.operator.base_llm_operator import BaseLLMOperator
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logger import Logger
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.monitor import Monitor
from utils.pagers.llm_split_pager import LLMSplitPager
from utils.printer import printer, Color, Printer
from utils.random_seed import seeding
from utils.structure import Structure
from utils.submission import Submission
from utils.timer import Timer

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

        self.print = printer[('MAIN', '·', Color.CYAN)]
        Printer.logger = Logger(self.exp.log)
        self.print('START TIME:', datetime.datetime.now())
        # print command line arguments
        self.print(' '.join(sys.argv))
        self.print(json.dumps(Obj.raw(self.config), indent=4))

        Setting.device = self.get_device()
        Setting.simple_dev = self.exp.policy.simple_dev

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

        self.m_optimizer: Optional[torch.optim.Optimizer] = None
        self.m_scheduler = None

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
        if cuda in ['-1', -1] or cuda is False:
            self.print('choose cpu')
            return 'cpu'
        if isinstance(cuda, int) or isinstance(cuda, str):
            self.print(f'User select cuda {cuda}')
            return f"cuda:{cuda}"
        return GPU.auto_choose(torch_format=True)

    def log_interval(self, epoch, step, loss):
        self.print[f'epoch {epoch}'](f'step {step}, loss {loss:.4f}')

    def log_epoch(self, epoch, results):
        line = ', '.join([f'{metric} {results[metric]:.4f}' for metric in results])
        self.print[f'epoch {epoch}'](line)

    def train(self) -> int:
        monitor_kwargs = Obj.raw(self.exp.store)

        dev_func = self.dev
        if Setting.simple_dev:
            monitor_kwargs['maximize'] = False
            dev_func = self.simple_evaluate
            self.print('activate simple dev mode')

        monitor = Monitor(
            save_dir=self.exp.dir,
            **monitor_kwargs,
        )

        train_steps = len(self.config_manager.sets.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        loader = self.config_manager.get_loader(Phases.train).train()
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.recommender.train()
            loader.train()
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

            dev_results, monitor_metric = dev_func()
            self.log_epoch(epoch, dev_results)

            state_dict = dict(
                model=self.recommender.state_dict(),
                optimizer=self.m_optimizer.state_dict(),
                scheduler=self.m_scheduler.state_dict(),
            )
            early_stop = monitor.push(
                epoch=epoch,
                metric=monitor_metric,
                state_dict=state_dict,
            )
            if early_stop == -1:
                return monitor.get_best_epoch()

        self.print('Training Ended')
        monitor.export()

        return monitor.get_best_epoch()

    def dev(self):
        assert self.exp.store.metric
        loader = self.config_manager.get_loader(Phases.dev).eval()

        results = self.evaluate(loader, metrics=[self.exp.store.metric])
        return results, results[self.exp.store.metric]

    def test_fake(self):
        self.recommender.eval()
        loader = self.config_manager.get_loader(Phases.test).test()
        loader.dataset.timer.clear()

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

    def mind_large_evaluate(self, loader):
        self.recommender.eval()

        # group_series = submission.depot.data[self.config_manager.column_map.group_col].tolist()
        # item_series = submission.depot.data[self.config_manager.column_map.candidate_col].tolist()
        # score_series = [random.random() for _ in range(len(submission.depot))]
        item_col, group_col = self.config_manager.column_map.candidate_col, self.config_manager.column_map.group_col
        score_series, col_series = self.base_evaluate(loader, cols=[item_col, group_col])
        item_series, group_series = col_series[item_col], col_series[group_col]
        # item_series = [v[0] for v in item_series]

        loader.dataset.timer.summarize()

        submission = Submission(
            depot=self.config_manager.depots[Phases.test],
            column_map=self.config_manager.column_map,
        )

        export_dir = submission.run(
            scores=score_series,
            groups=group_series,
            items=item_series,
            model_name=self.model.name,
        )

        self.print(f'export to {export_dir}')

    def test(self):
        loader = self.config_manager.get_loader(Phases.test).test()

        if self.config.mind_large_submission:
            return self.mind_large_evaluate(loader)

        results = self.evaluate(loader, metrics=self.exp.metrics)
        for metric in results:
            self.print(f'{metric}: {results[metric]:.4f}')

    def base_evaluate(self, loader, cols):
        score_series = torch.zeros(len(loader.dataset), dtype=torch.float32)
        col_series = {col: torch.zeros(len(loader.dataset), dtype=torch.long) for col in cols}

        index = 0
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            # timer.run('score')
            with torch.no_grad():
                scores = self.recommender(batch=batch)
                scores = scores.squeeze(1)
            # timer.run('score')

            batch_size = scores.size(0)
            # timer.run('extend')
            for col in cols:
                if batch[col].dim() == 2:
                    col_series[col][index:index + batch_size] = batch[col][:, 0]
                else:
                    col_series[col][index:index + batch_size] = batch[col]
            score_series[index:index + batch_size] = scores.cpu().detach()
            # score_series.extend(scores.cpu().detach())
            index += batch_size
            # timer.run('extend')

        # timer.summarize()
        # loader.dataset.timer.summarize()
        return score_series, col_series

    def evaluate(self, loader, metrics):
        pool = MetricPool.parse(metrics)
        self.recommender.eval()

        label_col, group_col = self.config_manager.column_map.label_col, self.config_manager.column_map.group_col
        score_series, col_series = self.base_evaluate(loader, cols=[label_col, group_col])
        label_series, group_series = col_series[label_col], col_series[group_col]

        results = pool.calculate(score_series, label_series, group_series)
        return results

    def simple_evaluate(self, **kwargs):
        loader = self.config_manager.get_loader(Phases.dev).eval()
        total_loss = Meaner()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.recommender(batch=batch)
            total_loss.add(loss.item())

        total_loss = total_loss.mean()
        return dict(loss=total_loss), total_loss

    def train_runner(self):
        if self.recommender.config.use_news_content and self.exp.policy.news_lr:
            self.print('split news encoder parameters')
            news_parameters, rec_parameters = self.recommender.parameter_split()
            self.m_optimizer = torch.optim.Adam(
                [
                    dict(
                        params=news_parameters,
                        lr=self.exp.policy.news_lr,
                    ), dict(
                        params=rec_parameters,
                        lr=self.exp.policy.lr,
                    )
                ],
                lr=self.exp.policy.lr
            )
        else:
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
        named_parameters = list(self.recommender.named_parameters())
        # filter out the parameters that don't require a gradient
        named_parameters = [(name, p) for (name, p) in named_parameters if p.requires_grad]
        # list of (name, parameter) pairs
        for (name, p) in named_parameters:
            self.print(name, p.data.shape)
        num_params = sum([p.numel() for (_, p) in named_parameters])
        # to a million
        num_params /= 1e6
        self.print(f'number of parameters: {num_params:.2f}M')

    def test_llm_layer_split(self):
        news_encoder = self.recommender.news_encoder  # type: BaseLLMOperator
        assert isinstance(news_encoder, BaseLLMOperator), 'llama operator not found'

        pager = LLMSplitPager(
            inputer=news_encoder.inputer,
            layers=Obj.raw(self.exp.store.layers),
            hidden_size=news_encoder.config.embed_hidden_size,
            contents=self.manager.doc_cache,
            model=news_encoder.get_all_hidden_states,
            page_size=self.exp.policy.batch_size,
        )

        pager.run()
        pager.store(self.exp.store.dir)

    def run(self):
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


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model', 'exp', 'embed'],
        default_args=dict(
            warmup=0,
            fast_eval=True,
            simple_dev=False,
            batch_size=64,
            acc_batch=1,
            lora=1,
            lora_r=32,
            lr=0.0001,
            news_lr=0.00001,
            mind_large_submission=False,
            hidden_size=64,
            epoch_batch=0,
            max_news_batch_size=0,
            fast_eval_batch_size=512,
        ),
        makedirs=[
            'exp.dir',
        ]
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
