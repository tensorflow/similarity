import math
import numpy as np
import numpy.ma as ma
from typing import Callable

from tensorflow_similarity.metrics import EvalMetric

# A function that takes the [num_query, num_neighbors] match mask and
# lookup_distnces and returns a [num_query, 1] boolean array indicating a
# correct label match between the query and the neighbors.
MatchFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _any_match_fn(match_mask: np.ndarray,
                  lookup_distances: np.ndarray) -> np.ndarray:
    _ = lookup_distances
    return np.any(match_mask, axis=1)


class MatchAccuracy(EvalMetric):
    """How many correct matches are returned for the given set pf parameters
    Probably the most important metric. Binary accuracy and match() is when
    k=1.

    num_matches / num_queries

    Accuracy is technically (TP+TN)/(TP+FP+TN+FN), but here we filter all
    queries above the distance threshold. In the case of binary matching,
    this makes all the TPs and FPs below the distance threshold and all the TNs
    and FNs above the distance threshold. As we are only concerned with the
    matches below the distance threshold, the accuracy simplifies to TP/(TP+FP)
    and is equivelent to the precision with respect to the unfiltered queries.
    However, we also want to consider the query coverage at the distance
    threshold, i.e., the predicted positive rate computed as
    (TP+FP)/(TP+FP+TN+FN). Therefore, we can take precision * query_coverage to
    produce a measure that capture the precision scaled by the query coverage.
    This simplifies down to the accuracy presented here, giving
    TP/(TP+FP+TN+FN).

    Attributes:
        match_fn: A function that takes the [num_queries, num_neighbors]
        np.ndarray match_mask and lookup_distances and returns a
        [num_queries, 1] match_indicator np.ndarray. This is used to indicate a
        positive label match for the query given a potentially filtered set of
        neighbor labels. By default this returns np.any(x, axis=1).

        name: Name associated with the metric object, e.g., precision@5

        canonical_name: The canonical name associated with metric,
        e.g., precision@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearset neighbor is
        considered a valid match.

        average: {'micro', 'macro'} Determines the type of averaging performed
        on the data.

            'micro': Calculates metrics globally over all data.

            'macro': Calculates metrics for each label and takes the unweighted
                     mean.
    """
    def __init__(self,
                 match_fn: MatchFn = _any_match_fn,
                 name: str = '',
                 distance_threshold: float = math.inf,
                 k: int = 1,
                 **kwargs) -> None:
        name = name if name else f'match_accuracy@{k}'

        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'match_accuracy'

        super().__init__(name=name,
                         distance_threshold=distance_threshold,
                         k=k,
                         **kwargs)

        self._match_fn = match_fn

    def compute(self,
                *,
                query_labels: np.ndarray,
                lookup_distances: np.ndarray,
                match_mask: np.ndarray,
                **kwargs) -> float:
        """Compute the metric

        Args:
            query_labels: A 1D array of the labels associated with the
            embedding queries.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighboor and a 0 indicates a mismatch.

            **kwargs: Additional compute args.

        Returns:
            metric results.
        """
        filtered_match_mask = ma.masked_values(
                match_mask,
                ~(lookup_distances <= self.distance_threshold)
        )
        k_slice = filtered_match_mask[:, :self.k]

        match_indicator = self._match_fn(k_slice.data, lookup_distances)

        if self.average == 'micro':
            match_accuracy = np.mean(match_indicator)
        elif self.average == 'macro':
            per_class_metrics = 0
            class_labels = np.unique(query_labels)
            for label in class_labels:
                idxs = np.argwhere(query_labels == label)
                per_class_metrics += np.mean(match_indicator[idxs])
            match_accuracy = np.divide(per_class_metrics, len(class_labels))
        else:
            raise ValueError(f'{self.average} is not a supported average '
                             'option')

        return match_accuracy
