from typing import Dict, Union

import torch
from sklearn.metrics import log_loss, roc_auc_score, ndcg_score, label_ranking_average_precision_score, f1_score


class Metric:
    name: str

    def calculate(self, scores: list, labels: list):
        pass


class LogLoss(Metric):
    name = 'LogLoss'

    def calculate(self, scores: list, labels: list):
        return log_loss(labels, scores)


class AUC(Metric):
    name = 'AUC'

    def calculate(self, scores: list, labels: list):
        return roc_auc_score(labels, scores)


class MRR(Metric):
    name = 'MRR'

    def calculate(self, scores: list, labels: list):
        return label_ranking_average_precision_score([labels], [scores])


class F1(Metric):
    name = 'F1'

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def calculate(self, scores: list, labels: list):
        scores = [int(score >= self.threshold) for score in scores]
        return f1_score(labels, scores)


class HitRatio(Metric):
    name = 'HitRatio'

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        labels[:self.n]
        # candidates_set = set(scores[:self.n])
        # interaction = candidates_set.intersection(set(labels))
        # return int(bool(interaction))


class Recall(Metric):
    name = 'Recall'

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        candidates_set = set(scores[:self.n])
        interaction = candidates_set.intersection(set(labels))
        return len(interaction) * 1.0 / self.n


class NDCG(Metric):
    name = 'NDCG'

    def __init__(self, n):
        self.n = n

    def calculate(self, scores: list, labels: list):
        return ndcg_score([labels], [scores], k=self.n)


class MetricPool:
    def __init__(self):
        self.pool = []
        self.metrics = dict()  # type: Dict[str, Metric]
        self.values = dict()  # type: Dict[tuple, Union[list, float]]
        self.max_n = -1

    def add(self, *metrics: Metric, ns=None):
        ns = ns or [None]

        for metric in metrics:
            self.metrics[metric.name] = metric

            for n in ns:
                if n and n > self.max_n:
                    self.max_n = n
                self.pool.append((metric.name, n))

    def init(self):
        self.values = dict()
        for metric_name, n in self.pool:
            self.values[(metric_name, n)] = []

    def push(self, candidates, ground_truth):
        for metric_name, n in self.values:
            if n and len(ground_truth) < n:
                continue

            self.values[(metric_name, n)].append(self.metrics[metric_name].calculate(
                scores=candidates,
                labels=ground_truth,
                n=n,
            ))

    def export(self):
        for metric_name, n in self.values:
            self.values[(metric_name, n)] = torch.tensor(
                self.values[(metric_name, n)], dtype=torch.float).mean().item()
