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

from typing import Tuple

import tensorflow as tf

from .classification_match import ClassificationMatch
from tensorflow_similarity.types import FloatTensor, IntTensor, BoolTensor


class MatchNearest(ClassificationMatch):
    """Match metrics for labels at k=1."""

    def __init__(self,
                 name: str = 'nearest',
                 **kwargs) -> None:

        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'match_nearest'

        super().__init__(name=name, **kwargs)

    def compute_match_indicators(self,
                                 query_labels: IntTensor,
                                 lookup_labels: IntTensor,
                                 lookup_distances: FloatTensor
                                 ) -> Tuple[BoolTensor, BoolTensor]:
        """Compute the indicator tensor.

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of BoolTensors:
                label_match: A len(query_labels x 1 boolean tensor. True if
                the match label == query label, False otherwise.

                dist_mask: A len(query_labels) x len(distance_thresholds)
                boolean tensor. True if the distance of the jth match <= the
                kth distance threshold.
        """
        if tf.rank(query_labels) == 1:
            query_labels = tf.expand_dims(query_labels, axis=-1)

        ClassificationMatch._check_shape(
                query_labels,
                lookup_labels,
                lookup_distances
        )

        # A 1D BoolTensor [len(query_labels), 1]
        label_match = tf.math.equal(
                query_labels,
                lookup_labels[:, :1]
        )
        # A 2D BoolTensor [len(lookup_distance), len(self.distance_thresholds)]
        dist_mask = tf.math.less_equal(
                lookup_distances[:, :1],
                self.distance_thresholds,
        )

        return label_match, dist_mask
