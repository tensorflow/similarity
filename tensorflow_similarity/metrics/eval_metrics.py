from abc import abstractmethod
from abc import ABC
import math
from typing import Sequence, Union

import numpy as np

from tensorflow_similarity.types import Lookup


class EvalMetric(ABC):
    """Abstract base class for computing evaluation metrics.

    Attributes:
        name: Name associated with the metric object, e.g., recall@5

        canonical_name: The canonical name associated with metric,
        e.g., recall@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearset neighbor is
        considered a valid match.

        average: {'micro'} Determines the type of averaging performed over the
        queries.

            'micro': Calculates metrics globally over all queries.

            'macro': Calculates metrics for each label and takes the unweighted
                     mean.
    """
    def __init__(self,
                 name: str,
                 canonical_name: str,
                 k: int = 1,
                 distance_threshold: float = math.inf,
                 average: str = 'micro') -> None:
        self.k = k
        self.distance_threshold = distance_threshold
        self.canonical_name = canonical_name
        self.name = self._suffix_name(name, k, distance_threshold)
        self.average = average

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "%s:%s" % (self.canonical_name, self.name)

    def get_config(self):
        return {
            "name": str(self.name),
            "canonical_name": str(self.canonical_name),
            "k": int(self.k),
            "distance_threshold": float(self.distance_threshold)
        }

    @staticmethod
    def from_config(config):
        metric = make_metric(config['canonical_name'])
        metric.name = config['name']
        metric.k = config['k']
        metric.distance_threshold = config['distance_threshold']
        return metric

    @abstractmethod
    def compute(self,
                *,
                query_labels: np.ndarray,
                lookup_labels: np.ndarray,
                lookup_distances: np.ndarray,
                match_mask: np.ndarray,
                lookups: Sequence[Sequence[Lookup]]) -> float:
        """Compute the metric

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighboor and a 0 indicates a mismatch.

            lookups: A 2D collection of Lookup results where the jth row is the
            k neighbors for the jth query.

        Returns:
            metric results.
        """

    def _suffix_name(self,
                     name: str,
                     k: int = 1,
                     distance: float = math.inf) -> str:
        "Suffix metric name with k and distance if needed"
        if k and k >= 1:
            name = "%s@%d" % (name, k)

        if distance and distance != 0.5:
            name = "%s:%f" % (name, distance)

        return name


def make_metric(metric: Union[str, EvalMetric]) -> EvalMetric:
    """Covert metric from str name to object if needed.

    Args:
        metric: EvalMetric() or metric name.

    Raises:
        ValueError: metric name is invalid.

    Returns:
        EvalMetric: Instantiated metric if needed.
    """
    # ! Metrics must be non-instantiated.
    METRICS_ALIASES = {
    }

    if isinstance(metric, EvalMetric):
        return metric
    elif isinstance(metric, str):
        if metric.lower() in METRICS_ALIASES:
            return METRICS_ALIASES[metric.lower()]
        else:
            raise ValueError('Unknown metric name:', metric, ' typo?')
    else:
        raise ValueError('metrics must be a str or a Evalmetric Object')
