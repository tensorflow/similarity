# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Mapping, Optional, Type, Union
import typing

import tensorflow as tf

from tensorflow_similarity.types import IntTensor
from tensorflow_similarity.types import BoolTensor
from .bndcg import BNDCG
from .retrieval_metric import RetrievalMetric
from .map_at_k import MapAtK
from .precision_at_k import PrecisionAtK
from .recall_at_k import RecallAtK


def compute_match_mask(query_labels: IntTensor,
                       lookup_labels: IntTensor) -> BoolTensor:
    """Compute a boolean mask (indicator function) marking the TPs in the results.

    Args:
        query_labels: A 1D tensor of the labels associated with the queries.

        lookup_labels: A 2D tensor where the jth row is the labels associated
        with the set of k neighbors for the jth query.

    Returns:
        A 2D boolean tensor indicating which lookups match the label of their
        associated query.
    """
    if tf.rank(query_labels) == 1:
        query_labels = tf.expand_dims(query_labels, axis=-1)

    match_mask: BoolTensor = tf.math.equal(lookup_labels, query_labels)

    return match_mask


def make_retrieval_metric(metric: Union[str, RetrievalMetric],
                          k: Optional[int] = None,
                          distance_threshold: Optional[float] = None,
                          r: Optional[Mapping[int, int]] = None
                          ) -> RetrievalMetric:
    """Convert metric from str name to object if needed.

    Args:
        metric: RetrievalMetric() or metric name.

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

        r: A mapping from class id to the number of examples in the index,
        e.g., r[4] = 10 represents 10 indexed examples from class 4. Only
        required for the MAP metric.

    Raises:
        ValueError: metric name is invalid.

    Returns:
        RetrievalMetric: Instantiated metric if needed.
    """
    # ! Metrics must be non-instantiated.
    METRICS_ALIASES: Dict[str, Type[RetrievalMetric]] = {
        # recall
        "recall": RecallAtK,
        "recall@k": RecallAtK,
        # precision
        "precision": PrecisionAtK,
        "precision@k": PrecisionAtK,
        # MAP
        "map": MapAtK,
        "map@k": MapAtK,
        # binary ndcg
        "bndcg": BNDCG,
        "bndcg@k": BNDCG,
    }

    if isinstance(metric, str):
        if metric.lower() in METRICS_ALIASES:
            valid_metric: RetrievalMetric = METRICS_ALIASES[metric.lower()]()
        else:
            raise ValueError(f'Unknown metric name: {metric}, typo?')
    else:
        valid_metric = metric

    if k:
        valid_metric.k = k
    if distance_threshold:
        valid_metric.distance_threshold = distance_threshold
    if r and valid_metric.canonical_name == "map@k":
        # valid_matric must be MapAtK if r is not None
        # TODO(ovallis): Find a better way to support r in MapAtK without
        # changing the protoype for RetrievalMetric
        typing.cast(MapAtK, valid_metric).r = r

    return valid_metric
