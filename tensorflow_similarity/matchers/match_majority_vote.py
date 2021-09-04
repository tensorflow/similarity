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

from typing import Callable, Tuple

import tensorflow as tf

from .classification_match import ClassificationMatch
from tensorflow_similarity.types import FloatTensor, IntTensor

# A Distance aggregation function used to compute the agg distance over the K
# neighbors. This distance is used, along with the distance threshold, to
# accept or reject the match label.
DistAgg = Callable[[FloatTensor, int], FloatTensor]


class MatchMajorityVote(ClassificationMatch):
    """Match metrics for the most common label in a result set."""

    def __init__(self,
                 name: str = 'majority_vote',
                 dist_agg: DistAgg = tf.math.reduce_mean,
                 **kwargs) -> None:

        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'match_majority_vote'

        super().__init__(name=name, **kwargs)

        self._dist_agg = dist_agg

    def predict(self,
                lookup_labels: IntTensor,
                lookup_distances: FloatTensor
                ) -> Tuple[FloatTensor, FloatTensor]:
        """Compute the predicted labels and distances.

        Given a set of lookup labels and distances, derive the predicted labels
        associated with the queries.

        This strategy takes the majority label in the lookups of the jth row as
        the predicted label for the jth query. In the case of a tie, we take the
        predicted label closest to the query.

        Additionally, the distance is taken as the aggregate of the distances
        in the jth row of lookups. The aggregation function can be passed to
        the constructor as a callable, and is set to tf.math.reduce_mean by
        default.

        Args:
            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of FloatTensors:
                predicted_labels: A FloatTensor of shape [len(lookup_labels),
                1] where the jth row contains the label predicted for the jth
                query.

                predicted_distances: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the distance
                associated with the jth predicted label.
        """

        # TODO(ovallis): Add parallel for callback or inline evaluation.
        pred_labels = tf.map_fn(self._majority_vote, lookup_labels)
        pred_labels = tf.expand_dims(pred_labels, axis=-1)

        # Callable type requires positional args only. Here we assume the
        # signature to be _dist_agg(input_tensor, axis)
        agg_dist = self._dist_agg(lookup_distances, 1)
        agg_dist = tf.expand_dims(agg_dist, axis=-1)

        return pred_labels, agg_dist

    def _majority_vote(self, lookup_labels):
        labels, _, counts = tf.unique_with_counts(lookup_labels)
        majority = tf.argmax(counts)

        return tf.gather(labels, majority)
