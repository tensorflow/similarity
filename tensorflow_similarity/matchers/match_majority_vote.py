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

from tensorflow_similarity.types import FloatTensor, IntTensor

from .classification_match import ClassificationMatch


class MatchMajorityVote(ClassificationMatch):
    """Match metrics for the most common label in a result set."""

    def __init__(self, name: str = "majority_vote", **kwargs) -> None:

        if "canonical_name" not in kwargs:
            kwargs["canonical_name"] = "match_majority_vote"

        super().__init__(name=name, **kwargs)

    def derive_match(self, lookup_labels: IntTensor, lookup_distances: FloatTensor) -> Tuple[IntTensor, FloatTensor]:
        """Derive a match label and distance from a set of K neighbors.

        For each query, derive a single match label and distance given the
        associated set of lookup labels and distances.

        Args:
            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of FloatTensors:
                derived_labels: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the derived
                label for the jth query.

                derived_distances: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the distance
                associated with the jth derived label.
        """

        # TODO(ovallis): Add parallel for callback or inline evaluation.
        pred_labels = tf.map_fn(self._majority_vote, lookup_labels)
        pred_labels = tf.expand_dims(pred_labels, axis=-1)

        agg_dist = tf.math.reduce_mean(lookup_distances, 1)
        agg_dist = tf.expand_dims(agg_dist, axis=-1)

        return pred_labels, agg_dist

    def _majority_vote(self, lookup_labels):
        labels, _, counts = tf.unique_with_counts(lookup_labels)
        majority = tf.argmax(counts)

        return tf.gather(labels, majority)
