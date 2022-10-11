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

from typing import Mapping

import tensorflow as tf

from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor

from .retrieval_metric import RetrievalMetric


class MapAtK(RetrievalMetric):
    r"""Mean Average precision - mAP@K is computed as.

    $$
    mAP_i@K = \frac{\sum_{j = 1}^{K} {rel_i_j}\times{P_i@j}}{R}
    $$

    Where: K is the number of neighbors in the i_th query result set.
           P is the rolling precision over the i_th query result set.
           R is the cardinality of the target class.
           rel is the relevance mask (indicator function) for the i_th query.
           i represents the i_th query.
           j represents the j_th ranked query result.

    AP@K is biased towards the top ranked results and is a function of the rank
    (K), the relevancy mask (rel), and the number of indexed examples for the
    class (R). The denominator for the i_th query is set to the number of
    indexed examples (R) for the class associated with the i_th query.

    For example, if the index has has 100 embedded examples (R) of class 'a',
    and our query returns 50 results (K) where the top 10 results are all TPs,
    then the AP@50 will be 0.10; however, if instead the bottom 10 ranked
    results are all TPs, then the AP@50 will be much lower (0.012) because we
    apply a penalty for the 40 FPs that come before the relevant query results.

    This metric is useful when we want to ensure that the top ranked results
    are relevant to the query; however, it requires that we pass a mapping from
    the class id to the number of indexed examples for that class.

    Args:
        r: A mapping from class id to the number of examples in the index,
        e.g., r[4] = 10 represents 10 indexed examples from class 4.

        clip_at_r: If True, set precision at K values to 0.0 for all values
        with rank greater than the count defined in the r mapping, e.g., a
        result set of 10 values, for a query label with 7 indexed examples,
        will have the last 3 precision at K values set to 0.0. Use this to
        compute MAP@R. See section 3.2 of https://arxiv.org/pdf/2003.08505.pdf

        name: Name associated with the metric object, e.g., avg_precision@5

        canonical_name: The canonical name associated with metric, e.g.,
        avg_precision@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

        average: {'micro'} Determines the type of averaging performed over the
        queries.

        * 'micro': Calculates metrics globally over all queries.

        drop_closest_lookup: If True, remove the closest lookup before computing
        the metrics. This is used when the query set == indexed set.
    """

    def __init__(
        self,
        r: Mapping[int, int],
        clip_at_r: bool = False,
        name: str = "map",
        **kwargs,
    ) -> None:
        if kwargs.get("average") == "macro":
            raise ValueError("Mean Average Precision only supports micro averaging.")

        if "canonical_name" not in kwargs:
            kwargs["canonical_name"] = "map@k"

        super().__init__(name=name, **kwargs)
        self.r = r
        self.clip_at_r = clip_at_r

    def get_config(self):
        config = {
            "r": self.r,
            "clip_at_r": self.clip_at_r,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute(
        self,
        *,  # keyword only arguments see PEP-570
        query_labels: IntTensor,
        match_mask: BoolTensor,
        **kwargs,
    ) -> FloatTensor:
        """Compute the metric

        Args:
            query_labels: A 1D array of the labels associated with the
            embedding queries.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighbor and a 0 indicates a mismatch.

            **kwargs: Additional compute args

        Returns:
            A rank 0 tensor containing the metric.
        """
        self._check_shape(query_labels, match_mask)

        start_k = 0
        if self.drop_closest_lookup:
            start_k = 1
            self.r = {k: tf.math.maximum(1, v - 1) for k, v in self.r.items()}

        k_slice = tf.cast(match_mask[:, start_k : start_k + self.k], dtype="float")

        tp = tf.math.cumsum(k_slice, axis=1)
        p_at_k = tf.math.divide(tp, tf.range(1, self.k + 1, dtype="float"))
        p_at_k = tf.math.multiply(k_slice, p_at_k)

        if self.average == "micro":
            table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(self.r.keys()),
                    list(self.r.values()),
                    key_dtype=query_labels.dtype,
                    value_dtype=query_labels.dtype,
                ),
                default_value=-1,
            )
            class_counts = table.lookup(query_labels)

            if self.clip_at_r:
                query_masks = []
                result_set_size = tf.shape(p_at_k)[1]
                for cc in class_counts:
                    cc = tf.cast(cc, dtype=result_set_size.dtype)
                    num_zeros = tf.math.subtract(result_set_size, cc)
                    query_masks.append(
                        tf.concat([tf.ones([1, cc]), tf.zeros([1, num_zeros])], axis=1)
                    )

                mask = tf.concat(query_masks, axis=0)
                p_at_k = tf.math.multiply(p_at_k, mask)

            avg_p_at_k = tf.math.divide(
                tf.math.reduce_sum(p_at_k, axis=1),
                tf.cast(class_counts, dtype="float"),
            )

            avg_p_at_k = tf.math.reduce_mean(avg_p_at_k)
        else:
            raise ValueError(f"{self.average} is not a supported average option")

        result: FloatTensor = avg_p_at_k
        return result
