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

import tensorflow as tf

from .retrieval_metric import RetrievalMetric
from tensorflow_similarity.types import FloatTensor, IntTensor, BoolTensor


class BNDCG(RetrievalMetric):
    """Binary normalized discounted cumulative gain.

    This is normalized discounted cumulative gain where the relevance weights
    are binary, i.e., correct match or incorrect match.

    Attributes:
        name: Name associated with the metric object, e.g., precision@5

        canonical_name: The canonical name associated with metric,
        e.g., precision@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

        average: {'micro', 'macro'} Determines the type of averaging performed
        on the data.

            'micro': Calculates metrics globally over all data.

            'macro': Calculates metrics for each label and takes the unweighted
                     mean.
    """
    def __init__(self,
                 name: str = 'ndcg',
                 k: int = 5,
                 **kwargs) -> None:
        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'ndcg@k'

        super().__init__(name=name, k=k, **kwargs)

    def compute(self,
                *,
                query_labels: IntTensor,
                lookup_distances: FloatTensor,
                match_mask: BoolTensor,
                **kwargs) -> FloatTensor:
        """Compute the metric

        Args:
            query_labels: A 1D array of the labels associated with the
            embedding queries.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighbor and a 0 indicates a mismatch.

            **kwargs: Additional compute args.

        Returns:
            metric results.
        """
        dist_mask = tf.math.less_equal(
                lookup_distances,
                self.distance_threshold
        )

        k_slice = tf.math.multiply(
                tf.cast(match_mask, dtype='float'),
                tf.cast(dist_mask, dtype='float')
        )[:, :self.k]
        rank = tf.range(1, self.k+1, dtype='float')
        rank_weights = tf.math.divide(
                tf.math.log1p(rank),
                tf.math.log(2.0)
        )

        # the numerator is simplier here because we are using binary weights
        dcg = tf.divide(k_slice, rank_weights)
        dcg = tf.math.reduce_sum(dcg, axis=1)
        per_example_ndcg = tf.divide(
                dcg,
                tf.math.reduce_sum(rank_weights)
        )

        if self.average == 'micro':
            ndcg = tf.math.reduce_mean(per_example_ndcg)
        elif self.average == 'macro':
            per_class_metrics = 0
            class_labels = tf.unique(query_labels)[0]
            for label in class_labels:
                idxs = tf.where(query_labels == label)
                c_slice = tf.gather(per_example_ndcg, indices=idxs)
                per_class_metrics += tf.math.reduce_mean(c_slice)
            ndcg = tf.math.divide(per_class_metrics, len(class_labels))
        else:
            raise ValueError(f'{self.average} is not a supported average '
                             'option')

        result: FloatTensor = ndcg
        return result
