from typing import Dict, List, Sequence, Union

import numpy as np

from tensorflow_similarity.evaluator import Evaluator
from tensorflow_similarity._metrics import EvalMetric, make_metric
from tensorflow_similarity.types import Lookup


class MemoryEvaluator(Evaluator):
    """In memory indexed embedding evaluation."""

    def evaluate(self,
                 index_size: int,
                 metrics: Sequence[Union[str, EvalMetric]],
                 query_labels: Sequence[int],
                 lookups: Sequence[Sequence[Lookup]],
                 distance_rounding: int = 8
                 ) -> Dict[str, float]:

        # [nn[{'distance': xxx}, ]]
        # normalize metrics
        eval_metrics: List[EvalMetric] = [make_metric(m) for m in metrics]

        query_labels = np.ndarray(query_labels).reshape(-1, 1)

        lookup_labels = unpack_lookup_labels(lookups)
        lookup_dists = unpack_lookup_distances(lookups)
        match_mask = compute_match_mask(query_labels, lookup_labels)

        # compute metrics
        evaluation = {}
        for m in eval_metrics:
            evaluation[m.name] = m.compute(
                    query_labels,
                    lookup_labels,
                    lookup_dists,
                    match_mask,
            )

        return evaluation
