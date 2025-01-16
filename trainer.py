import torch
from pigmento import pnt

from loader.env import Env
from loader.symbols import Symbols
from tester import Tester
from utils import bars
from utils.config_init import CommandInit
from utils.meaner import Meaner
from utils.metrics import MetricPool
from utils.monitor import Monitor


class Trainer(Tester):
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

    def run(self):
        self.init_optimizer()
        self.init_scheduler()
        self.load()
        self.train()
        self.load(Env.path_hub.signature)
        self.test()


def get_configurations():
    return CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            embed='config/embed/null.yaml',
            exp='config/exp/default.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            item_page_size=64,
        ),
    ).parse()


if __name__ == '__main__':
    configuration = get_configurations()
    trainer = Trainer(config=configuration)
    trainer.run()
