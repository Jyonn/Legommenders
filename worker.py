import json
import os
import time

import torch
from oba import Obj
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.global_loader import GlobalLoader
from loader.global_setting import Setting
from task.base_batch import BaseBatch, HSeqBatch
from task.base_loss import BaseLoss, LossDepot
from task.base_task import BaseTask
from utils.config_init import ConfigInit
from utils.gpu import GPU
from utils.logger import Logger
from utils.monitor import Monitor
from utils.printer import printer, Color, Printer
from utils.structure import Structure


class Worker:
    def __init__(self, config):
        self.config = config
        self.data, self.model, self.exp = self.config.data, self.config.model, self.config.exp
        self.disable_tqdm = self.exp.policy.disable_tqdm

        Setting.device = self.get_device()

        self.print = printer[('MAIN', '·', Color.CYAN)]
        self.logging = Logger(self.config.data.store.log_file)
        Printer.logger = self.logging
        self.print(json.dumps(Obj.raw(self.config), indent=4))

        self.global_loader = GlobalLoader(
            data=self.data,
            model=self.model,
            exp=self.exp
        )
        self.model_container = self.global_loader.model_container
        self.task = self.global_loader.primary_task

        self.print(Structure.analyse_and_stringify(self.global_loader.train_set[0]))
        # self.abnormal_detector()

        self.static_modes = ['export', 'dev', 'test']
        self.in_static_modes = self.exp.mode in self.static_modes or self.exp.mode.startswith('test')

        if self.in_static_modes:
            self.m_optimizer = self.m_scheduler = None
        else:
            self.m_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.model_container.parameters()),
                lr=self.exp.policy.lr
            )
            self.m_scheduler = get_linear_schedule_with_warmup(
                self.m_optimizer,
                num_warmup_steps=self.exp.policy.n_warmup,
                num_training_steps=len(self.global_loader.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
            )

            self.print('training params')
            total_memory = 0
            for name, p in self.model_container.named_parameters():  # type: str, torch.Tensor
                total_memory += p.element_size() * p.nelement()
                if p.requires_grad:
                    self.print(name, p.data.shape)

    def _attempt_loading(self, path):
        load_path = os.path.join(self.data.store.save_dir, path)
        while True:
            self.print(f"load model from exp {load_path}")
            try:
                state_dict = torch.load(load_path, map_location=Setting.device)
                break
            except Exception as e:
                if not self.exp.load.wait_load:
                    raise e
                time.sleep(60)

        model_ckpt = state_dict['model']

        self.model_container.load_state_dict(model_ckpt, strict=not self.exp.load.relax_load)
        load_status = False
        if not self.in_static_modes and not self.exp.load.load_model_only:
            load_status = True
            self.m_optimizer.load_state_dict(state_dict['optimizer'])
            self.m_scheduler.load_state_dict(state_dict['scheduler'])
        self.print(f'load optimizer and scheduler: {load_status}')

    def attempt_loading(self):
        if self.exp.load.load_ckpt:
            self._attempt_loading(self.exp.load.load_ckpt)

    def get_device(self):
        cuda = self.config.cuda
        if cuda in [-1, False]:
            return 'cpu'
        if not cuda:
            return GPU.auto_choose(torch_format=True)
        return f"cuda:{cuda}"

    def log_interval(self, epoch, step, task: BaseTask, loss: BaseLoss):
        components = [f'step {step}', f'task {task.name}']
        for loss_name, loss_tensor in loss.get_loss_dict().items():
            if loss_name.endswith('loss') and isinstance(loss_tensor, torch.Tensor):
                components.append(f'{loss_name} {loss_tensor.item():.4f}')
        self.print[f'epoch {epoch}'](', '.join(components))

    def log_epoch(self, epoch, task: BaseTask, loss_depot: LossDepot):
        components = [f'task {task.name}']
        for loss_name, loss_value in loss_depot.table.items():
            components.append(f'{loss_name} {loss_value:.4f}')
        self.print[f'epoch {epoch}'](', '.join(components))

    def train(self):
        monitor = Monitor(
            save_dir=self.data.store.save_dir,
            **Obj.raw(self.exp.store)
        )

        train_steps = len(self.global_loader.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        loader = self.global_loader.get_dataloader(Setting.TRAIN).train()
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            # loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.model_container.train()

            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):  # type: int, BaseBatch
                task_output = self.model_container(
                    batch=batch,
                    task=self.task,
                )

                loss = self.task.calculate_loss(task_output, batch, model=self.model_container)
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
                            self.log_interval(epoch, step, self.task, loss)
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, self.task, loss)

            loss_depot = self.dev()
            self.log_epoch(epoch, self.task, loss_depot)

            state_dict = dict(
                model=self.model_container.state_dict(),
                optimizer=self.m_optimizer.state_dict(),
                scheduler=self.m_scheduler.state_dict(),
            )
            monitor.push(
                epoch=epoch,
                loss=loss_depot.table,
                state_dict=state_dict,
            )

        self.print('Training Ended')
        monitor.export()

    def dev(self, steps=None):
        self.model_container.eval()
        loader = self.global_loader.get_dataloader(Setting.DEV).eval()
        loss_depot = LossDepot()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                task_output = self.model_container(
                    batch=batch,
                    task=self.global_loader.primary_task,
                )

                loss = self.task.calculate_loss(task_output, batch, model=self.model_container)
                loss_depot.add(loss)

            if steps and step >= steps:
                break

        return loss_depot.summarize()

    def speed_test(self):
        samples = []
        for sample in tqdm(self.global_loader.train_set):
            samples.append(sample)
        del samples
        batches = []
        for batch in tqdm(self.global_loader.get_dataloader(Setting.TRAIN)):
            batches.append(batch)
        del batches

    def run(self):
        if self.exp.mode == 'train':
            self.train()
        elif self.exp.mode == 'dev':
            loss_depot = self.dev(10)
            self.log_epoch(0, self.task, loss_depot)
        elif self.exp.mode == 'speed_test':
            self.speed_test()


if __name__ == '__main__':
    configuration = ConfigInit(
        required_args=['data', 'model', 'exp'],
        makedirs=[
            'data.store.save_dir',
        ]
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
