from abc import abstractmethod
import tensorflow as tf
from typing import List, Union


class EvalMetric():

    def __init__(self,
                 direction: str,
                 name: str,
                 canonical_name: str,
                 k: int = None,
                 distance_threshold: float = None) -> None:
        if direction not in ['min', 'max']:
            raise ValueError('Unknown direction - must be in {min,max}')
        self.k = k
        self.distance_threshold = distance_threshold
        self.canonical_name = canonical_name
        self.name = self._suffix_name(name, k, distance_threshold)
        self.direction = direction

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "%s:%s" % (self.canonical_name, self.name)

    def get_config(self):
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "direction": self.direction,
            "k": self.k,
            "distance_threshold": self.distance_threshold
        }

    @staticmethod
    def from_config(self, config):
        metric = get_metric_from_name(config['canonical_name'])
        metric.name = config['name']
        metric.k = config['k']
        metric.distance_threshold = config['distance_threshold']
        return metric

    @abstractmethod
    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> Union[int, float]:
        pass
        # match_ranks: rank 0 is unmatched, rank1 is first neighbopor, rank2 ...
        # match_distance: distance 0 means infinite, rest is match distances

    def _suffix_name(self, name: str, k: int = None, distance: float = None) -> str:
        "Suffix metric name with k and distance if needed"
        if k and k > 1:
            name = "%s@%d" % (name, k)

        if distance and distance != 0.5:
            name = "%s:%f" % (name, distance)

        return name

    def filter_ranks(self,
                     match_ranks: List[int],
                     match_distances: List[float],
                     min_rank: int = 1,
                     max_rank: int = None,
                     distance: float = None) -> List[int]:
        """Filter match ranks to only keep matches between `min_rank`
        and `max_rank` below a give distance.

        Args:
            match_ranks (List[int]): [description]
            min_rank (int, optional): Minimal rank to keep (inclusive).
            Defaults to 1.

            max_rank (int, optional): Max rank to keep (inclusive).
            Defaults to None. If None will keep all ranks above `min_rank`

        Returns:
            List[int]: filtered ranks as a dense array with missing elements
            removed. len(filtered_ranks) <= len(match_ranks)
        """
        filtered_ranks = []
        if not max_rank:
            max_rank = max(match_ranks)

        if not distance:
            distance = max(match_distances)

        for idx, r in enumerate(match_ranks):
            if min_rank <= r <= max_rank and match_distances[idx] <= distance:
                filtered_ranks.append(r)
        if not len(filtered_ranks):
            return [-1]
        return filtered_ranks


class MeanRank(EvalMetric):

    def __init__(self, name='mean_rank') -> None:
        super().__init__(direction='min',
                         canonical_name='mean_rank',
                         name=name)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> float:

        # remove unmatched elements (rank 0)
        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=max_k)
        return float(tf.reduce_mean(matches))


class MinRank(EvalMetric):

    def __init__(self, name='min_rank') -> None:
        super().__init__(direction='min', canonical_name='min_rank', name=name)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> int:

        # remove unmatched elements (rank 0)
        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=max_k)
        return int(tf.reduce_min(matches))


class MaxRank(EvalMetric):

    def __init__(self, name='max_rank') -> None:
        super().__init__(direction='min', canonical_name='max_rank', name=name)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> int:

        # remove unmatched elements (rank 0)
        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=max_k)
        return int(tf.reduce_max(matches))


class Accuracy(EvalMetric):
    """How many correct matches are returned for the given paramters
    probably the most important metric. Accuracy can be at k=1 when
    """

    def __init__(self,
                 distance_threshold: float = 0.5,
                 k: int = 1,
                 name='accuracy') -> None:
        super().__init__(direction='max',
                         name=name,
                         canonical_name='accuracy',
                         k=k,
                         distance_threshold=distance_threshold)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> float:

        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=self.k,
                                    distance=self.distance_threshold)

        return len(matches) / len(targets_labels)


class Precision(EvalMetric):
    """ Compute the precision of the matches.

    Notes: precision computation for similarity is different from
    the classic version and also tricky to get right so please refers
    to the tf.similarity paper for a complete discussion of how it works.

    Precision formula: `true_positives / (true_positives + false_positives)`


    Reference: TBD
    """

    def __init__(self,
                 distance_threshold: float = 0.5,
                 k: int = 1,
                 name='precision') -> None:

        super().__init__(direction='max',
                         name=name,
                         canonical_name='precision',
                         k=k,
                         distance_threshold=distance_threshold)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> float:

        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=self.k,
                                    distance=self.distance_threshold)

        true_positives = len(matches)
        false_positives = num_matched - true_positives
        return true_positives / (true_positives + false_positives)


class Recall(EvalMetric):
    """Computing matcher recall at k for a given distance threshold

    Recall formula: `true_positive / (true_positive + false_negative)`
    """

    def __init__(self,
                 distance_threshold: float = 0.5,
                 k: int = 1,
                 name='recall') -> None:
        super().__init__(direction='max',
                         name=name,
                         canonical_name='recall',
                         k=k,
                         distance_threshold=distance_threshold)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> float:

        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=self.k,
                                    distance=self.distance_threshold)

        true_positives = len(matches)
        # targets_labels = true positive + false negative
        return true_positives / len(targets_labels)


class F1Score(EvalMetric):
    """Compute the F1 score, also known as balanced F-score or F-measure at K
    below a distance threshold.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


    f1_score formula is: `2 * (precision * recall) / (precision + recall)`

    """

    def __init__(self,
                 distance_threshold: float = 0.5,
                 k: int = 1,
                 name='f1_score') -> None:
        super().__init__(direction='max',
                         name=name,
                         canonical_name='f1_score',
                         k=k,
                         distance_threshold=distance_threshold)

    def compute(self,
                max_k: int,
                targets_labels: List[int],
                num_matched: int,
                num_unmatched: int,
                index_size: int,
                match_ranks: List[int],
                match_distances: List[float],
                matches_labels: List[List[int]]) -> float:

        matches = self.filter_ranks(match_ranks,
                                    match_distances,
                                    max_rank=self.k,
                                    distance=self.distance_threshold)

        true_positives = len(matches)
        false_positives = num_matched - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / len(targets_labels)

        return 2 * (precision * recall) / (precision + recall)


def get_metric_from_name(metric_name) -> EvalMetric:
    # ! Metrics must be non-instanciated.
    METRICS_ALIASES = {
        # accuracy
        "accuracy": Accuracy(),
        "acc": Accuracy(),
        # recall
        "recall": Recall(),
        # precision
        "precision": Precision(),
        # f1 score
        "f1_score": F1Score(),
        "f1": F1Score(),
        # mean rank
        "mean_rank": MeanRank(),
        "meanrank": MeanRank(),
        # max rank
        "max_rank": MaxRank(),
        "maxrank": MaxRank(),
        # min rank
        "min_rank": MinRank(),
        "minrank": MinRank(),
    }
    if metric_name.lower() in METRICS_ALIASES:
        return METRICS_ALIASES[metric_name.lower()]
    else:
        raise ValueError('Unknown metric name:', metric_name, ' typo?')


def build_metrics(metrics: List[Union[str, EvalMetric]]) -> List[EvalMetric]:
    """Convert a list of mixed metrics name and object to a list
    of EvalMetrics

    Args:
        metrics (List[Union[str, EvalMetric]]): Metric and Metric names

    Returns:
        List[EvalMetric]: Metrics objects
    """
    eval_metrics = []
    for m in metrics:
        if isinstance(m, EvalMetric):
            eval_metrics.append(m)
        elif isinstance(m, str):
            eval_metrics.append(EvalMetric.get_metric_from_name(m))
        else:
            raise ValueError('metrics must be a str or a Evalmetric Object')
    return eval_metrics
