import json
import os

import torch
from oba import Obj


class Monitor:
    def __init__(
            self,
            interval=None,
            monitor=None,
            ckpt_path=None,
            task=None,
            top=None,
            epoch_skip=None,
            early_stop=None,
    ):
        self.interval = interval
        self.candidates = []
        self.monitor = monitor
        self.ckpt_path = ckpt_path
        self.task = task
        self.top = top
        self.epoch_skip = epoch_skip
        self.early_stop = early_stop

    def remove_checkpoint(self, epoch):
        epoch_path = os.path.join(self.ckpt_path, 'epoch_{}.bin'.format(epoch))
        os.system(f'rm {epoch_path}')

    def store_checkpoint(self, epoch, state_dict):
        epoch_path = os.path.join(self.ckpt_path, 'epoch_{}.bin'.format(epoch))
        torch.save(state_dict, epoch_path)

        self.step_export()

    def push(self, epoch, loss_depots: dict, state_dict):
        if self.epoch_skip and epoch < self.epoch_skip:
            return

        if self.interval:
            if (epoch + 1) % self.interval == 0:
                self.store_checkpoint(epoch, state_dict)
            return

        if len(loss_depots) == 1:
            loss = list(loss_depots.values())[0]
        else:
            loss = loss_depots[self.task]
        loss = Obj(loss)
        self.candidates.append((epoch, loss))

        stay = [True] * len(self.candidates)

        for ia in range(len(self.candidates)):
            for ib in range(len(self.candidates)):
                if ia == ib or not stay[ia] or not stay[ib]:
                    continue
                a, b = self.candidates[ia][1], self.candidates[ib][1]
                if eval(self.monitor):
                    stay[ib] = False

        removing_checkpoints = []
        for i in range(len(self.candidates) - 1):
            if not stay[i]:
                removing_checkpoints.append(self.candidates[i][0])

        if self.top:
            for i in removing_checkpoints[-self.top:]:
                stay[i] = True
            removing_checkpoints = removing_checkpoints[:-self.top]
        for checkpoint in removing_checkpoints:
            self.remove_checkpoint(checkpoint)

        self.candidates = [self.candidates[i] for i in range(len(self.candidates)) if stay[i]]

        if not stay[-1]:
            if self.early_stop:
                for i in range(len(self.candidates))[::-1]:
                    if stay[i]:
                        if epoch - self.candidates[i][0] > self.early_stop:
                            raise ValueError('Early Stop')
                        return
            return

        self.store_checkpoint(epoch, state_dict)

    def step_export(self):
        candidates = list(map(lambda x: x[0], self.candidates))
        export_path = os.path.join(self.ckpt_path, 'candidates.json')
        json.dump(candidates, open(export_path, 'w'))

    def export(self):
        if self.top:
            for candidate in self.candidates[:-self.top]:
                self.remove_checkpoint(candidate[0])
            self.candidates = self.candidates[-self.top:]
        self.step_export()
