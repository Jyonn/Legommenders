import copy
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from loader.env import Env
from trainer import Tester, Trainer, get_configurations
from utils.function import get_signature


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.linear.weight, 0.1)

    def forward(self, batch):
        prediction = self.linear(batch['feature'])
        return nn.functional.mse_loss(prediction, batch['label'])


class TrainingScheduleTest(unittest.TestCase):
    def make_trainer(self, accumulate_batch=3, epoch_batch=0):
        trainer = Trainer.__new__(Trainer)
        trainer.exp = SimpleNamespace(
            policy=SimpleNamespace(
                epoch=2, epoch_batch=epoch_batch, accumulate_batch=accumulate_batch,
                n_warmup=0, check_interval=0,
            ),
            store=SimpleNamespace(patience=5, metric='GAUC'),
        )
        features = torch.linspace(0.1, 0.9, 9).unsqueeze(1)
        labels = 2 * features
        loader = DataLoader(
            [dict(feature=x, label=y) for x, y in zip(features, labels)], batch_size=2,
        )
        trainer.manager = SimpleNamespace(
            get_train_loader=lambda: loader, cm=SimpleNamespace(label_col='label'),
            setup=Mock(),
        )
        trainer.legommender = RegressionModel()
        trainer.optimizer = torch.optim.SGD(trainer.legommender.parameters(), lr=0.2)
        trainer.disable_tqdm = True
        trainer.dev = Mock(return_value=({'GAUC': 0.5}, 0.5))
        trainer.log_epoch = Mock()
        trainer.save = Mock()
        return trainer, features, labels

    def test_matches_combined_batches_and_steps_before_evaluation(self):
        trainer, features, labels = self.make_trainer()
        reference = copy.deepcopy(trainer.legommender)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.2)
        weights_at_validation = []
        for step, samples in enumerate([slice(0, 6), slice(6, 9)] * 2):
            optimizer.param_groups[0]['lr'] = 0.2 * (1 - step / 4)
            optimizer.zero_grad()
            reference(dict(feature=features[samples], label=labels[samples])).backward()
            optimizer.step()
            if step in (1, 3):
                weights_at_validation.append(reference.linear.weight.detach().clone())

        evaluated_steps = []

        def evaluate(_bar):
            torch.testing.assert_close(
                trainer.legommender.linear.weight, weights_at_validation[len(evaluated_steps)],
            )
            evaluated_steps.append(trainer.scheduler.last_epoch)
            return {'GAUC': 0.5}, 0.5

        trainer.dev.side_effect = evaluate
        with patch.object(Env, 'simple_dev', False):
            trainer.init_scheduler()
            trainer.train()
        self.assertEqual(evaluated_steps, [2, 4])
        self.assertEqual(trainer.scheduler.get_last_lr(), [0.0])
        self.assertIsNone(trainer.legommender.linear.weight.grad)
        trainer.save.assert_called_once()

    def test_epoch_limits_are_shared_by_the_loop_and_scheduler(self):
        for limit, batches in [(1, 1), (-2, 2), (20, 5)]:
            with self.subTest(limit=limit):
                trainer, _, _ = self.make_trainer(accumulate_batch=1, epoch_batch=limit)
                with (
                    patch.object(Env, 'simple_dev', False),
                    patch.object(trainer.legommender, 'forward', wraps=trainer.legommender.forward) as forward,
                ):
                    trainer.init_scheduler()
                    trainer.train()
                self.assertEqual(forward.call_count, batches * 2)
                self.assertEqual(trainer.scheduler.last_epoch, batches * 2)
                self.assertEqual(trainer.scheduler.get_last_lr(), [0.0])

    def test_default_yaml_resolves_accumulation_from_command(self):
        args = {
            'data': 'config/data/mind.yaml', 'model': 'config/model/naml.yaml', 'batch_size': 8,
        }
        self.assertEqual(get_configurations(dict(args)).exp.policy.accumulate_batch, 1)
        self.assertEqual(get_configurations(dict(args, accumulate_batch=3)).exp.policy.accumulate_batch, 3)

    def test_fast_mode_changes_only_the_enabled_run_signature(self):
        args = {
            'data': 'config/data/mind.yaml', 'model': 'config/model/naml.yaml', 'batch_size': 8,
        }
        default, off, on = [get_configurations(dict(args, **options)) for options in (
            {}, {'fast': False}, {'fast': True},
        )]
        with patch.object(Tester, '__init__', return_value=None):
            for config in (default, off, on):
                Trainer(config)

        def signature(config):
            return get_signature(config.data, config.embed, config.model, config.exp)

        self.assertEqual(signature(default), signature(off))
        self.assertNotEqual(signature(default), signature(on))
        self.assertIs(on.exp.policy.fast, True)

    def test_fast_mode_rejects_cpu(self):
        trainer, _, _ = self.make_trainer()
        trainer.config = SimpleNamespace(cuda=-1)
        trainer.exp.policy.fast = True
        with self.assertRaisesRegex(ValueError, 'native BF16 support'):
            trainer.get_device()

    def test_fast_mode_checks_the_selected_cuda_device(self):
        trainer, _, _ = self.make_trainer()
        trainer.exp.policy.fast = True
        with (
            patch.object(Tester, 'get_device', return_value='cuda:1'),
            patch.object(torch.cuda, 'is_available', return_value=True),
            patch.object(torch.cuda, 'device') as select_device,
            patch.object(torch.cuda, 'is_bf16_supported', return_value=True) as supported,
        ):
            self.assertEqual(trainer.get_device(), torch.device('cuda:1'))
        select_device.assert_called_once_with(torch.device('cuda:1'))
        supported.assert_called_once_with(including_emulation=False)

    def test_fast_autocast_is_scoped_to_the_training_forward(self):
        trainer, _, _ = self.make_trainer(accumulate_batch=3)
        trainer.exp.policy.fast = True
        active = False
        forwards = []
        backward = []

        @contextmanager
        def autocast(device, *, dtype):
            nonlocal active
            self.assertEqual((device, dtype), ('cuda', torch.bfloat16))
            active = True
            try:
                yield
            finally:
                active = False

        def record_forward(_module, _inputs, output):
            forwards.append(active)
            output.register_hook(lambda _grad: backward.append(active))

        def evaluate(_bar):
            self.assertFalse(active)
            return {'GAUC': 0.5}, 0.5

        trainer.legommender.register_forward_hook(record_forward)
        trainer.dev.side_effect = evaluate
        with patch.object(Env, 'simple_dev', False), patch.object(torch, 'autocast', autocast):
            trainer.init_scheduler()
            trainer.train()
        self.assertEqual(forwards, [True] * 10)
        self.assertEqual(backward, [False] * 10)


if __name__ == '__main__':
    unittest.main()
