import torch
from tqdm import tqdm

from loader.mode.base_mode import BaseMode


class EvaluateMode(BaseMode):
    load_model = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.disable_tqdm = self.controller.exp.policy.disable_tqdm

    def work(self, *args, loader, cols, **kwargs):
        score_series = torch.zeros(len(loader.dataset), dtype=torch.float32)
        col_series = {col: torch.zeros(len(loader.dataset), dtype=torch.long) for col in cols}

        index = 0
        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
            with torch.no_grad():
                scores = self.legommender(batch=batch)
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
