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

from collections.abc import Mapping

import tensorflow as tf

from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor

from .retrieval_metric import RetrievalMetric


class PrecisionAtK(RetrievalMetric):
    r"""Precision@K is computed as.

    $$
    P_i@k = \frac{TP_i}{TP_i+FP_i} = \frac{\sum_{j = 1}^{k} {rel_i_j}}{K}
    $$

    Where: K is the number of neighbors in the i_th query result set.
           rel is the relevance mask (indicator function) for the i_th query.
           i represents the i_th query.
           j represents the j_th ranked query result.

    P@K is unordered and does not take into account the rank of the TP results.

    This metric is useful when we are interested in evaluating the embedding
    within the context of a kNN classifier or as part of a clustering method.

    Args:
        r: An optional mapping from class id to the number of examples in the
        index, e.g., r[4] = 10 represents 10 indexed examples from class 4.
        If None, the value for k will be used instead.

        clip_at_r: If True, set precision at K values to 0.0 for all values
        with rank greater than the count defined in the r mapping, e.g., a
        result set of 10 values, for a query label with 7 indexed examples,
        will have the last 3 precision at K values set to 0.0. Use this to
        compute R Precision. See section 3.2 of https://arxiv.org/pdf/2003.08505.pdf

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
        r: Mapping[int, int] | None = None,
        clip_at_r: bool = False,
        name: str = "precision",
        **kwargs,
    ) -> None:
        if "canonical_name" not in kwargs:
            kwargs["canonical_name"] = "precision@k"

        super().__init__(name=name, **kwargs)
        self.r = r
        self.clip_at_r = clip_at_r

    @property
    def name(self) -> str:
        if self.clip_at_r:
            return f"{self._name}"

        return super().name

    def get_config(self):
        config = {
            "r": self.r,
            "clip_at_r": self.clip_at_r,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _generate_class_count_table(self, class_labels, dtype):
        r = self.r
        if r is None:
            r = {int(lbl): self.k for lbl in class_labels}

        if self.drop_closest_lookup:
            r = {k: tf.math.maximum(0, v - 1) for k, v in r.items()}

        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(r.keys()),
                list(r.values()),
                key_dtype=dtype,
                value_dtype=dtype,
            ),
            default_value=0,
        )

    def _get_matches(self, x):
        return x

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

            **kwargs: Additional compute args.

        Returns:
            A rank 0 tensor containing the metric.
        """
        self._check_shape(query_labels, match_mask)
        class_labels = tf.unique(tf.reshape(query_labels, (-1)))[0]
        start_k = 0

        if self.drop_closest_lookup:
            start_k = 1

        k_slice = tf.cast(match_mask[:, start_k : self.k], dtype="float")
        matches = self._get_matches(k_slice)

        class_count_table = self._generate_class_count_table(class_labels, query_labels.dtype)
        class_counts = class_count_table.lookup(query_labels)

        if self.clip_at_r:
            elems = (matches, tf.expand_dims(class_counts, axis=1))
            # for each row i in matches, reduce_sum the first class_count[i] values and divide by
            # class_count[i]
            per_example = tf.map_fn(
                lambda x: tf.math.divide_no_nan(
                    tf.math.reduce_sum(x[0][: x[1][0]]),
                    tf.cast(x[1], dtype="float"),
                ),
                elems,
                fn_output_signature="float",
            )
        else:
            per_example = tf.math.divide_no_nan(
                tf.math.reduce_sum(matches, axis=1),
                tf.cast(class_counts, dtype="float"),
            )

        if self.average == "micro":
            avg_res = tf.math.reduce_mean(per_example)
        elif self.average == "macro":
            per_class_metrics = 0
            for label in class_labels:
                idxs = tf.where(query_labels == label)
                c_slice = tf.gather(per_example, indices=idxs)
                per_class_metrics += tf.math.reduce_mean(c_slice)
            avg_res = tf.math.divide_no_nan(per_class_metrics, len(class_labels))
        else:
            raise ValueError(f"{self.average} is not a supported average " "option")
        result: FloatTensor = avg_res
        return result
