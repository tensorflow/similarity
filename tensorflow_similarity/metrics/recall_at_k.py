import numpy as np

from tensorflow_similarity.metrics import EvalMetric


class RecallAtK(EvalMetric):
    r"""The metric learning version of Recall@K.

    A query is counted as a positive when ANY lookup in top K match the query
    class, 0 otherwise.

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
                 name: str = '',
                 k: int = 1,
                 **kwargs) -> None:
        name = name if name else f'recall@{k}'

        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'recall@k'

        super().__init__(name=name, k=k, **kwargs)

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

            **kwargs: Additional compute args.

        Returns:
            metric results.
        """
        k_slice = match_mask[:, :self.k]
        match_indicator = np.any(k_slice, axis=1)

        if self.average == 'micro':
            recall_at_k = np.mean(match_indicator)
        elif self.average == 'macro':
            per_class_metrics = 0
            class_labels = np.unique(query_labels)
            for label in class_labels:
                idxs = np.argwhere(query_labels == label)
                per_class_metrics += np.mean(match_indicator[idxs])
            recall_at_k = np.divide(per_class_metrics, len(class_labels))
        else:
            raise ValueError(f'{self.average} is not a supported average '
                             'option')

        return recall_at_k
