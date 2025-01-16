import warnings
from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Union, List

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score, ndcg_score, label_ranking_average_precision_score, f1_score


class Metric:
    name: str
    group: bool
    minimize: bool

    def calculate(self, scores: list, labels: list) -> Union[int, float]:
        pass

    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)

    def __str__(self):
        return self.name


class LogLoss(Metric):
    name = 'LogLoss'
    group = False
    minimize = True

    def calculate(self, scores: list, labels: list):
        return log_loss(labels, scores)


class AUC(Metric):
    name = 'AUC'
    group = False
    minimize = False

    def calculate(self, scores: list, labels: list):
        return roc_auc_score(labels, scores)


class GAUC(AUC):
    name = 'GAUC'
    group = True
    minimize = False


class LRAP(Metric):
    name = 'LRAP'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list):
        return label_ranking_average_precision_score([labels], [scores])


class MRR0(Metric):
    name = 'MRR0'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list):
        ranked_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)

        for rank, idx in enumerate(ranked_indices, start=1):  # type: int, int
            if labels[idx] == 1:
                return 1 / rank
        return 0


class MRR(Metric):
    name = 'MRR'
    group = True
    minimize = False

    def calculate(self, scores: list, labels: list):
        order = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        y_true = [labels[i] for i in order]
        rr_score = [y_true[i] / (i + 1) for i in range(len(y_true))]
        return sum(rr_score) / sum(y_true)


class F1(Metric):
    name = 'F1'
    group = False
    minimize = False

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def calculate(self, scores: list, labels: list):
        scores = [int(score >= self.threshold) for score in scores]
        return f1_score(labels, scores)

    def __str__(self):
        return f'{self.name}@{self.threshold}'


class HitRatio(Metric):
    name = 'HitRatio'
    group = True
    minimize = False

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        return int(1 in labels[:self.n])
        # candidates_set = set(scores[:self.n])
        # interaction = candidates_set.intersection(set(labels))
        # return int(bool(interaction))

    def __str__(self):
        return f'{self.name}@{self.n}'


class Recall(Metric):
    name = 'Recall'
    group = True
    minimize = False

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        return sum(labels[:self.n]) * 1.0 / sum(labels)
        # candidates_set = set(scores[:self.n])
        # interaction = candidates_set.intersection(set(labels))
        # return len(interaction) * 1.0 / self.n

    def __str__(self):
        return f'{self.name}@{self.n}'


class NDCG(Metric):
    name = 'NDCG'
    group = True
    minimize = False

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        return ndcg_score([labels], [scores], k=self.n)

    def __str__(self):
        return f'{self.name}@{self.n}'


class MetricPool:
    metric_list = [LogLoss, AUC, GAUC, F1, Recall, NDCG, HitRatio, LRAP, MRR, MRR0]
    metric_dict = {m.name.upper(): m for m in metric_list}

    def __init__(self, metrics):
        self.metrics = metrics  # type: List[Metric]
        self.values = OrderedDict()  # type: Dict[str, Union[list, float]]
        self.group = False

        for metric in self.metrics:
            self.values[str(metric)] = []
            self.group = self.group or metric.group

    @classmethod
    def parse(cls, metrics_config):
        metrics = []
        for m in metrics_config:
            at = m.find('@')
            argument = []
            if at > -1:
                m, argument = m[:at], [int(m[at+1:])]
            if m.upper() not in MetricPool.metric_dict:
                raise ValueError(f'Metric {m} not found')
            metric = MetricPool.metric_dict[m.upper()](*argument)
            if isinstance(metric, MRR):
                warnings.warn('Following existing recommendation repositories, '
                              'the implementation of MRR is not the same as the original one. '
                              'To get the original MRR, use MRR0 instead.')
            metrics.append(metric)
        return cls(metrics)

    def calculate(self, scores, labels, groups, group_worker=5):
        if not self.metrics:
            return {}

        df = pd.DataFrame(dict(groups=groups, scores=scores, labels=labels))

        groups = None
        if self.group:
            groups = df.groupby('groups')

        for metric in self.metrics:
            if not metric.group:
                self.values[str(metric)] = metric(
                    scores=scores,
                    labels=labels,
                )
                continue

            tasks = []
            pool = Pool(processes=group_worker)
            for g in groups:
                group = g[1]
                g_labels = group.labels.tolist()
                g_scores = group.scores.tolist()
                tasks.append(pool.apply_async(metric, args=(g_scores, g_labels)))
            pool.close()
            pool.join()
            values = [t.get() for t in tasks]
            self.values[str(metric)] = torch.tensor(values, dtype=torch.float).mean().item()
        return self.values

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    @classmethod
    def is_minimize(cls, metric: str):
        if isinstance(metric, Metric):
            return metric.minimize
        assert isinstance(metric, str)
        metric = metric.split('@')[0]
        return cls.metric_dict[metric].minimize
