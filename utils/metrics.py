"""
metrics.py

This module contains:

1. A *Metric* base-class that defines a unified interface for all evaluation
   metrics used in this project.
2. Concrete implementations of point-wise metrics (LogLoss, AUC, …) and
   group-wise / ranking metrics (GAUC, HitRatio@K, Recall@K, NDCG@K, …).
3. A *MetricPool* class that acts as a light-weight metric manager:
      – parsing a user supplied metric configuration,
      – feeding predictions/labels to every metric,
      – (optionally) aggregating metrics on a per-group basis, and
      – returning the final scalar results.

The code purposefully avoids heavyweight libraries (e.g. `pytorch-lightning`)
and instead keeps the evaluation logic self-contained and customisable.
"""

import warnings
from collections import OrderedDict
from multiprocessing import Pool
from typing import Dict, Union, List, Sequence, Any

import pandas as pd
import torch
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    ndcg_score,
    label_ranking_average_precision_score,
    f1_score,
)


# =============================================================================
#                               Metric interface
# =============================================================================
class Metric:
    """
    Abstract base-class.

    Sub-classes only need to implement *calculate* and specify three class
    attributes:

        name     : Human-readable metric name (e.g. "AUC")
        group    : Whether the metric must be evaluated *within* groups
                   (e.g. user-wise) and then averaged.
        minimize : If `True`, a *lower* value is considered better
                   (useful for early-stopping or hyper-parameter search).
    """

    name: str            # e.g. "AUC@10"
    group: bool          # if True → calculate per group then average
    minimize: bool       # if True → smaller is better

    # ---------------------------------------------------------------------
    # API every derived metric has to implement
    # ---------------------------------------------------------------------
    def calculate(self, scores: Sequence[float], labels: Sequence[int]) -> Union[int, float]:
        """Override in sub-class to compute the metric value."""
        raise NotImplementedError

    # Make metric objects *callable* for convenience ---------------------------------
    def __call__(self, *args, **kwargs) -> Union[int, float]:
        return self.calculate(*args, **kwargs)

    # Pretty string representation ---------------------------------------------------
    def __str__(self):
        return self.name


# =============================================================================
#                           Point-wise classification metrics
# =============================================================================
class LogLoss(Metric):
    """
    Binary cross-entropy (a.k.a. log-loss).
    """
    name = "LogLoss"
    group = False
    minimize = True

    def calculate(self, scores, labels):
        return log_loss(labels, scores)


class AUC(Metric):
    """
    Area Under ROC Curve.
    """
    name = "AUC"
    group = False
    minimize = False

    def calculate(self, scores, labels):
        return roc_auc_score(labels, scores)


class GAUC(AUC):
    """
    *Grouped* AUC (a.k.a. User-AUC).
    Implementation is identical to AUC but we set `group=True` so that
    the computation is performed per-group then averaged.
    """
    name = "GAUC"
    group = True
    minimize = False


class LRAP(Metric):
    """
    Label Ranking Average Precision (micro averaged).
    """
    name = "LRAP"
    group = True
    minimize = False

    def calculate(self, scores, labels):
        return label_ranking_average_precision_score([labels], [scores])


# =============================================================================
#                               Ranking metrics
# =============================================================================
class MRR0(Metric):
    """
    *Original* Mean Reciprocal Rank where only the FIRST relevant item is
    considered (returns 0 if there is none).
    """
    name = "MRR0"
    group = True
    minimize = False

    def calculate(self, scores, labels):
        ranked_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)

        for rank, idx in enumerate(ranked_indices, start=1):  # rank starts at 1
            if labels[idx] == 1:
                return 1 / rank
        return 0


class MRR(Metric):
    """
    Modified MRR that is commonly used in recommender system repositories.
    NOTE:
        This differs from the *original* definition and may yield a higher
        value when multiple relevant items exist.  A warning is emitted in
        MetricPool.parse() to make the user aware of this fact.
    """
    name = "MRR"
    group = True
    minimize = False

    def calculate(self, scores, labels):
        order = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        y_true = [labels[i] for i in order]               # labels in ranked order
        rr_score = [y_true[i] / (i + 1) for i in range(len(y_true))]
        return sum(rr_score) / sum(y_true)                # mean over positives


class F1(Metric):
    """
    Threshold-based F1 score for binary classification.
    """
    name = "F1"
    group = False
    minimize = False

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def calculate(self, scores, labels):
        # Convert probabilistic scores to 0/1 predictions
        preds = [int(score >= self.threshold) for score in scores]
        return f1_score(labels, preds)

    def __str__(self):
        # e.g. "F1@0.7"
        return f"{self.name}@{self.threshold}"


class HitRatio(Metric):
    """
    HitRatio@K — did we rank at least one positive item inside the top-K?
    """
    name = "HitRatio"
    group = True
    minimize = False

    def __init__(self, n: int):
        self.n = n

    def calculate(self, scores, labels):
        # Sort by score descending, then check the first K labels
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        return int(1 in labels[: self.n])

    def __str__(self):
        return f"{self.name}@{self.n}"


class Recall(Metric):
    """
    Recall@K — fraction of *all* positives that appear in the top-K.
    """
    name = "Recall"
    group = True
    minimize = False

    def __init__(self, n: int):
        self.n = n

    def calculate(self, scores, labels):
        scores, labels = zip(*sorted(zip(scores, labels), key=lambda x: x[0], reverse=True))
        return sum(labels[: self.n]) * 1.0 / sum(labels)

    def __str__(self):
        return f"{self.name}@{self.n}"


class NDCG(Metric):
    """
    Normalized Discounted Cumulative Gain @K.
    """
    name = "NDCG"
    group = True
    minimize = False

    def __init__(self, n: int):
        self.n = n

    def calculate(self, scores, labels):
        return ndcg_score([labels], [scores], k=self.n)

    def __str__(self):
        return f"{self.name}@{self.n}"


# =============================================================================
#                             Metric Pool / Manager
# =============================================================================
class MetricPool:
    """
    A thin wrapper that handles a *list* of Metric objects:

        pool = MetricPool.parse(["AUC", "NDCG@10", "HitRatio@5"])
        results = pool(scores, labels, groups)      # -> OrderedDict

    Features
    --------
    • Automatic type inference and construction via `parse`.
    • Parallel computation for expensive *group-wise* metrics.
    • Keeps track of whether any metric is group-wise so that only one
      groupby() pass over the data is needed.
    """

    # Registry of *all* available metric classes
    metric_list = [LogLoss, AUC, GAUC, F1, Recall, NDCG, HitRatio, LRAP, MRR, MRR0]
    metric_dict = {m.name.upper(): m for m in metric_list}

    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics                                   # instantiated metric objects
        self.values = OrderedDict()                              # metric_name -> float
        self.group = False                                       # does *any* metric require grouping?

        for metric in self.metrics:
            self.values[str(metric)] = []                        # placeholder
            self.group = self.group or metric.group

    # -------------------------------------------------------------------------
    # Building a MetricPool from a list of user supplied strings
    # -------------------------------------------------------------------------
    @classmethod
    def parse(cls, metrics_config: Sequence[str]) -> "MetricPool":
        """
        Convert a list like ["AUC", "NDCG@10", "F1@0.7"] into Metric objects.

        `@` is used as a separator between the metric name and its argument(s).
        """
        metrics: List[Metric] = []

        for m in metrics_config:
            at = m.find("@")
            argument: List[Any] = []
            if at > -1:
                # Split "NDCG@10" -> "NDCG", ["10"]
                m, argument = m[:at], [int(m[at + 1 :])]

            # Validate the metric name
            if m.upper() not in MetricPool.metric_dict:
                raise ValueError(f"Metric {m} not found")

            metric_cls = MetricPool.metric_dict[m.upper()]
            metric = metric_cls(*argument)

            # Warn about the non-standard MRR implementation
            if isinstance(metric, MRR):
                warnings.warn(
                    "Following existing recommendation repositories, "
                    "the implementation of MRR is *not* the original one. "
                    "To get the original definition, use MRR0 instead."
                )

            metrics.append(metric)

        return cls(metrics)

    # -------------------------------------------------------------------------
    # Main entry: compute all metrics
    # -------------------------------------------------------------------------
    def calculate(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        groups: Sequence[Any],
        group_worker: int = 5,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        scores  : predicted scores / probabilities.
        labels  : ground-truth binary labels (0 / 1).
        groups  : group identifier for each sample (e.g. user_id).
        group_worker : number of parallel processes for per-group evaluation.

        Returns
        -------
        OrderedDict(str -> float)
            Final scalar metric values.
        """
        if not self.metrics:
            return {}

        # Build a Pandas DataFrame for convenient group-by operations
        df = pd.DataFrame(dict(groups=groups, scores=scores, labels=labels))

        group_df = None
        if self.group:
            group_df = df.groupby("groups")

        # ------------------------------------------------------------------
        # Iterate over every metric and populate self.values
        # ------------------------------------------------------------------
        for metric in self.metrics:
            if not metric.group:
                # Point-wise metric: we can compute over the whole dataset
                self.values[str(metric)] = metric(scores=scores, labels=labels)
                continue

            # Group-wise metric – compute per group then average
            tasks = []
            pool = Pool(processes=group_worker)

            for g in group_df:
                group = g[1]                         # (group_id, group_dataframe)
                g_labels = group.labels.tolist()
                g_scores = group.scores.tolist()
                tasks.append(pool.apply_async(metric, args=(g_scores, g_labels)))

            pool.close()
            pool.join()

            values = [t.get() for t in tasks]
            # Convert to tensor for convenient mean() then back to Python float
            self.values[str(metric)] = torch.tensor(values, dtype=torch.float).mean().item()

        return self.values

    # Make the pool itself callable --------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Helper util to check optimisation direction ------------------------------
    # -------------------------------------------------------------------------
    @classmethod
    def is_minimize(cls, metric: Union[str, Metric]) -> bool:
        """
        Quick utility to learn whether a particular metric should be minimized.

        Accepts both a Metric object or its string representation (possibly
        with arguments, e.g. "NDCG@10").
        """
        if isinstance(metric, Metric):
            return metric.minimize

        assert isinstance(metric, str)
        metric = metric.split("@")[0]            # strip arguments
        return cls.metric_dict[metric.upper()].minimize
