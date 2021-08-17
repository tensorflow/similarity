from typing import Mapping

import numpy as np

from tensorflow_similarity.metrics import EvalMetric


class MeanAvgPrecisionAtK(EvalMetric):
    r"""Mean Average precision (mAP) @K is computed as.

               k
              ===
              \    rel   . P @j
              /       ij    i
              ===
             j = 1
    mAP @k = ------------------
       i           R

    Where: K is the number of neighbors in the i_th query result set.
           P is the rolling precision over the i_th query result set.
           R is the cardinality of the target class.
           rel is the relevance mask (indicator function) for the i_th query.
           i represents the i_th query.
           j represents the j_th ranked query result.

    AP@K is biased towards the top ranked results and is a function of both the
    rank (K) and the class size (R). This metric will be equal to the recall
    when all the top results are on contiguous block of TPs.

    For example, if the embedding dataset has 100 examples (R) of class 'a',
    and our query returns 50 results (K) where the top 10 results are all TPs,
    then the AP@50 will be 0.10; however, if instead the bottom 10 ranked
    results are all TPs, then the AP@50 will be much lower (0.012) because we
    apply a penalty for the 40 FPs that come before the relevant query results.

    This metric is useful when we want to ensure that the top ranked results
    are relevant to the query.

    Attributes:
        r: A mapping from class id to the number of examples in the index,
        e.g., r[4] = 10 represents 10 indexed examples from class 4

        name: Name associated with the metric object, e.g., avg_precision@5

        canonical_name: The canonical name associated with metric, e.g.,
        avg_precision@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearset neighbor is
        considered a valid match.

        average: {'micro'} Determines the type of averaging performed over the
        queries.

            'micro': Calculates metrics globally over all queries.
    """
    def __init__(self,
                 r: Mapping[int, int],
                 name: str = '',
                 k: int = 1,
                 average: str = 'micro',
                 **kwargs) -> None:
        if average == 'macro':
            raise ValueError('Mean Average Precision only supports micro '
                             'averaging.')

        for label, count in r.items():
            if count < k:
                raise ValueError(f'Class {label} has {count} examples in the '
                                 f'index but K is set to {k}. The number of '
                                 'indexed examples for each class must be '
                                 'greater than or equal to K.')

        name = name if name else f'avg_precision@{k}'

        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'avg_precision@k'

        super().__init__(name=name, k=k, average=average, **kwargs)
        self.r = r

    def compute(self,
                *,
                query_labels: np.ndarray,
                match_mask: np.ndarray,
                **kwargs) -> float:
        """Compute the metric

        Args:
            query_labels: A 1D array of the labels associated with the
            embedding queries.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighboor and a 0 indicates a mismatch.

            **kwargs: Additional compute args

        Returns:
            metric results.
        """

        mask_slice = match_mask[:, :self.k]
        tp = np.cumsum(mask_slice, axis=1)
        p_at_k = np.divide(
                tp,
                np.arange(1, self.k+1)
        )
        masked_p_at_k = mask_slice * p_at_k

        if self.average == 'micro':
            class_counts = np.array([self.r[label] for label in query_labels])
            avg_p_at_k = np.divide(
                    np.sum(masked_p_at_k, axis=1),
                    class_counts
            )

            avg_p_at_k = np.mean(avg_p_at_k)
        else:
            raise ValueError(f'{self.average} is not a supported average '
                             'option')

        return avg_p_at_k
