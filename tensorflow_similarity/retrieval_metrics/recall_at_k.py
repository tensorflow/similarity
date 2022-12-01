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
from __future__ import annotations

import tensorflow as tf

from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor

from .retrieval_metric import RetrievalMetric


class RecallAtK(RetrievalMetric):
    """The metric learning version of Recall@K.

    A query is counted as a positive when ANY lookup in top K match the query
    class, 0 otherwise.

    Args:
        name: Name associated with the metric object, e.g., recall@5

        canonical_name: The canonical name associated with metric,
        e.g., recall@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

        average: {'micro'} Determines the type of averaging performed over the
        queries.

        * 'micro': Calculates metrics globally over all queries.
        * 'macro': Calculates metrics for each label and takes the unweighted
        mean.

        drop_closest_lookup: If True, remove the closest lookup before computing
        the metrics. This is used when the query set == indexed set.
    """

    def __init__(self, name: str = "recall", **kwargs) -> None:
        if "canonical_name" not in kwargs:
            kwargs["canonical_name"] = "recall@k"

        super().__init__(name=name, **kwargs)

    def compute(
        self,
        *,  # keyword only arguments see PEP-570
        query_labels: IntTensor,
        match_mask: BoolTensor,
        **kwargs,
    ) -> FloatTensor:
        """Compute the metric

        Args:
            query_labels: A 1D tensor of the labels associated with the
            embedding queries.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighbor and a 0 indicates a mismatch.
            k
            **kwargs: Additional compute args.

        Returns:
            A rank 0 tensor containing the metric.
        """
        self._check_shape(query_labels, match_mask)

        start_k = 1 if self.drop_closest_lookup else 0
        k_slice = match_mask[:, start_k : start_k + self.k]

        match_indicator = tf.math.reduce_any(k_slice, axis=1)
        match_indicator = tf.cast(match_indicator, dtype="float")

        if self.average == "micro":
            recall_at_k = tf.math.reduce_mean(match_indicator)
        elif self.average == "macro":
            per_class_metrics = 0
            class_labels = tf.unique(tf.reshape(query_labels, (-1)))[0]
            # TODO(ovallis): potential slowness.
            for label in class_labels:
                idxs = tf.where(query_labels == label)
                c_slice = tf.gather(match_indicator, indices=idxs)
                per_class_metrics += tf.math.reduce_mean(c_slice)
            recall_at_k = tf.math.divide(per_class_metrics, len(class_labels))
        else:
            raise ValueError(f"{self.average} is not a supported average " "option")
        result: FloatTensor = recall_at_k
        return result
