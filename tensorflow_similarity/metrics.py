from abc import abstractmethod
import tensorflow as tf
from typing import List, Union


class EvalMetric():

    def __init__(self, direction: str, name: str) -> None:
        self.name = name
        self.direction = direction

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
        return filtered_ranks


class MeanRank(EvalMetric):

    def __init__(self, name='mean_rank') -> None:
        super().__init__(direction='min', name=name)

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
        matches = self.filter_ranks(match_ranks, match_distances, max_rank=max_k)
        return float(tf.reduce_mean(matches))


class MinRank(EvalMetric):

    def __init__(self, name='min_rank') -> None:
        super().__init__(direction='min', name=name)

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
        matches = self.filter_ranks(match_ranks, match_distances, max_rank=max_k)
        print(matches)
        return int(tf.reduce_min(matches))


class MaxRank(EvalMetric):

    def __init__(self, name='max_rank') -> None:
        super().__init__(direction='min', name=name)

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
        matches = self.filter_ranks(match_ranks, match_distances, max_rank=max_k)
        return int(tf.reduce_max(matches))


class Accuracy(EvalMetric):
    """How many correct matches are returned for the given paramters
    probably the most important metric. Accuracy can be at k=1 when
    """

    def __init__(self,
                 distance_threshold: float = 0.5,
                 k: int = 1,
                 name='accuracy') -> None:
        self.k = k
        self.distance_threshold = distance_threshold
        name = self._suffix_name(name, k, distance_threshold)
        super().__init__(direction='max', name=name)

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
        self.k = k
        self.distance_threshold = distance_threshold
        name = self._suffix_name(name, k, distance_threshold)
        super().__init__(direction='max', name=name)

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
        self.k = k
        self.distance_threshold = distance_threshold
        name = self._suffix_name(name, k, distance_threshold)
        super().__init__(direction='max', name=name)

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
        self.k = k
        self.distance_threshold = distance_threshold
        name = self._suffix_name(name, k, distance_threshold)
        super().__init__(direction='max', name=name)

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
