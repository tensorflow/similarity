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


class BNDCG(RetrievalMetric):
    r"""Binary normalized discounted cumulative gain.

    This is normalized discounted cumulative gain where the relevancy weights
    are binary, i.e., either a correct match or an incorrect match.

    The NDCG is a score between [0,1] representing the rank weighted results.
    The DCG represents the sum of the correct matches weighted by the log2 of
    the rank and is normalized by the 'ideal DCG'. The IDCG is computed as the
    match_mask, sorted descending, weighted by the log2 of the post sorting rank
    order. This metric takes into account both the correctness of the match and
    the position.

    The normalized DCG is computed as:

    $$
    nDCG_{p} = \frac{DCG_{p}}{IDCG_{p}}
    $$

    The DCG is computed for each query using the match_mask as:

    $$
    DCG_{p} = \sum_{i=1}^{p} \frac{match_mask_{i}}{\log_{2}(i+1)}
    $$

    The IDCG uses the same equation but sorts the match_mask descending
    along axis=-1.

    Additionally, all positive matches with a distance above the threshold are
    set to 0, and the closest K matches are taken.

    Args:
        name: Name associated with the metric object, e.g., precision@5

        canonical_name: The canonical name associated with metric,
        e.g., precision@K

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

    def __init__(
        self,
        name: str = "ndcg",
        **kwargs,
    ) -> None:
        if "canonical_name" not in kwargs:
            kwargs["canonical_name"] = "ndcg@k"

        super().__init__(name=name, **kwargs)

    def compute(
        self,
        *,  # keyword only arguments see PEP-570
        query_labels: IntTensor,
        lookup_distances: FloatTensor,
        match_mask: BoolTensor,
        **kwargs,
    ) -> FloatTensor:
        """Compute the metric

        Computes the binary NDCG. The query labels are only used when the
        averaging is set to "macro".

        Args:
            query_labels: A 1D array of the labels associated with the
            embedding queries.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighbor and a 0 indicates a mismatch.

        Returns:
            A rank 0 tensor containing the metric.
        """
        self._check_shape(query_labels, match_mask)

        if tf.shape(lookup_distances)[0] != tf.shape(query_labels)[0]:
            raise ValueError(
                "The number of lookup distance rows must equal the number "
                "of query labels. Number of lookup distance rows is "
                f"{tf.shape(lookup_distances)[0]} but the number of query "
                f"labels is {tf.shape(query_labels)[0]}."
            )

        dist_mask = tf.math.less_equal(lookup_distances, self.distance_threshold)

        start_idx = 1 if self.drop_closest_lookup else 0
        k_slice = tf.math.multiply(
            tf.cast(match_mask, dtype="float"),
            tf.cast(dist_mask, dtype="float"),
        )[:, start_idx : start_idx + self.k]

        rank = tf.range(1, self.k + 1, dtype="float")
        rank_weights = tf.math.divide(tf.math.log1p(rank), tf.math.log(2.0))

        # the numerator is simplier here because we are using binary weights
        dcg = tf.math.reduce_sum(k_slice / rank_weights, axis=1)

        # generate the "ideal ordering".
        ideal_ordering = tf.sort(k_slice, direction="DESCENDING", axis=1)
        idcg = tf.math.reduce_sum(ideal_ordering / rank_weights, axis=1)

        per_example_ndcg = tf.math.divide_no_nan(dcg, idcg)

        if self.average == "micro":
            ndcg = tf.math.reduce_mean(per_example_ndcg)
        elif self.average == "macro":
            per_class_metrics = 0
            class_labels = tf.unique(tf.reshape(query_labels, (-1)))[0]
            for label in class_labels:
                idxs = tf.where(query_labels == label)
                c_slice = tf.gather(per_example_ndcg, indices=idxs)
                per_class_metrics += tf.math.reduce_mean(c_slice)
            ndcg = tf.math.divide(per_class_metrics, len(class_labels))
        else:
            raise ValueError(f"{self.average} is not a supported average " "option")

        result: FloatTensor = ndcg
        return result
