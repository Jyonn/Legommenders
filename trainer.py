import psutil
import torch
from pigmento import pnt
from unitok import JsonHandler

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
    server: Server

    def prepare_live_experiment(self):
        if not self.config.session:
            return

        self.server = Server.auto_auth()
        experiment = self.server.get_experiment_info(session=self.config.session)
        experiment = ExperimentBody(experiment.body)
        if experiment.signature != Env.ph.signature:
            pnt(f"Signature mismatch: {Env.ph.signature} != {experiment.signature}, the live experiment will be terminated.")
            raise ValueError(f"Signature mismatch")
        if experiment.seed != self.config.seed:
            pnt(f"Seed mismatch: {self.config.seed} != {experiment.seed}, the live experiment will be terminated.")
            raise ValueError(f"Seed mismatch")
        if experiment.is_completed:
            pnt(f"Experiment {Env.ph.signature} is already completed, the live experiment will be terminated.")
            raise ValueError("Experiment is already completed")
        if experiment.pid is not None:
            if psutil.pid_exists(experiment.pid):
                pnt(f"Experiment {Env.ph.signature} is already running, the live experiment will be terminated.")
                raise ValueError("Experiment is already running")
        self.server.register_experiment(session=self.config.session)

    def simple_evaluate(self, dev_bar: bars.Bar):
        loader = self.manager.get_dev_loader()
        meaner = Meaner()

        for step, batch in enumerate(bar := dev_bar(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.legommender(batch=batch)
            bar.set_postfix_str(f'loss: {meaner(loss.item()):.4f}')

        return dict(loss=meaner.mean), meaner.mean

    def dev(self, dev_bar: bars.Bar):
        assert self.exp.store.metric
        loader = self.manager.get_dev_loader()

        results = self.evaluate(loader, metrics=[self.exp.store.metric], bar=dev_bar)
        return results, results[self.exp.store.metric]

    def train(self):
        monitor = Monitor(
            patience=self.exp.store.patience,
            minimize=Env.simple_dev or MetricPool.is_minimize(self.exp.store.metric)
        )

        dev_func = self.simple_evaluate if Env.simple_dev else self.dev

        train_steps = len(self.manager.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        accumulate_batch = self.exp.policy.accumulate_batch or 1

        check_interval = self.exp.policy.check_interval
        if check_interval and check_interval < 0:
            check_interval = max(train_steps // (-check_interval), 1)

        meaner = Meaner()
        loader = self.manager.get_train_loader()
        self.optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch):
            self.legommender.train()
            self.manager.setup(Symbols.train)
            # for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            for step, batch in enumerate(bar := bars.TrainBar(epoch=epoch)(loader, disable=self.disable_tqdm)):
                loss = self.legommender(batch=batch)
                loss.backward()

                bar.set_postfix_str(f'loss: {meaner(loss.item()):.4f}')

                accumulate_step += 1
                if accumulate_step == accumulate_batch:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0

                if check_interval and (step + 1) % check_interval == 0:
                    self.log_interval(epoch, step, meaner.mean)

                if self.exp.policy.epoch_batch:
                    if self.exp.policy.epoch_batch < 0:  # step part
                        if step > max(train_steps // (-self.exp.policy.epoch_batch), 1):
                            break
                    else:
                        if step > self.exp.policy.epoch_batch:
                            break

            dev_bar = bars.DevBar(epoch=epoch, train_loss=meaner.mean)
            dev_results, monitor_metric = dev_func(dev_bar)
            self.log_epoch(epoch, dev_results)

            action = monitor.push(monitor_metric)
            if action is Symbols.stop:
                pnt('Early stop')
                break
            elif action is Symbols.best:
                self.save()

        pnt('Training Ended')

    def test(self):
        results = super().test()
        self.complete_live_experiment(results)

    @staticmethod
    def get_pured_log():
        # with open(Env.ph.log_path, 'rb') as f:
        #     log_bin = f.read()
        log_bin = io.file_load(Env.ph.log_path, binary=True)

        # Clean up progress lines: if there's a \r in a line, keep only what's after the last \r
        lines = log_bin.split(b'\n')
        cleaned_lines = []
        for line in lines:
            if b'\r' in line:
                # Keep content after the last carriage return
                line = line[line.rfind(b'\r') + 1:]
            cleaned_lines.append(line)

        log_bin = b'\n'.join(cleaned_lines)
        return log_bin.decode('utf-8', errors='replace')

    def complete_live_experiment(self, results):
        if not self.config.session:
            return

        log = self.get_pured_log()

        performance = JsonHandler.dumps(results)
        self.server.complete_experiment(
            session=self.config.session,
            log=log,
            performance=performance
        )
        pnt(f"Experiment {Env.ph.signature} is completed and uploaded to the lego server.")

    def run(self):
        self.init_optimizer()
        self.init_scheduler()
        self.load()
        self.train()
        self.load(Env.ph.signature)
        self.test()


def get_configurations(kwargs=None):
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


if __name__ == '__main__':
    configuration = get_configurations()
    trainer = Trainer(config=configuration)
    trainer.run()
