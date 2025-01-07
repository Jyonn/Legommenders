import torch
from pigmento import pnt
from tqdm import tqdm

from loader.env import Env
from loader.symbols import Symbols
from tester import Tester
from utils.config_init import CommandInit
from utils.metrics import MetricPool
from utils.monitor import Monitor


class Trainer(Tester):
    def simple_evaluate(self):
        loader = self.manager.get_dev_loader()
        losses = []

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                loss = self.legommender(batch=batch)
            losses.append(loss.item())

        # total_loss = total_loss.mean()
        total_loss = sum(losses) / len(losses)
        return dict(loss=total_loss), total_loss

    def dev(self):
        assert self.exp.store.metric
        loader = self.manager.get_dev_loader()

        results = self.evaluate(loader, metrics=[self.exp.store.metric])
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

        loader = self.manager.get_train_loader()
        self.optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch):
            self.legommender.train()
            self.manager.setup(Symbols.train)
            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
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


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            embed='config/embed/null.yaml',
            exp='config/exp/default.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            item_page_size=64,
        ),
    ).parse()

    trainer = Trainer(config=configuration)
    trainer.run()
