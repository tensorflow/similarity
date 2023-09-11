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
"""Inner product similarity computation functions for embeddings."""
from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from tensorflow_similarity.types import FloatTensor

from .distance import Distance


@tf.keras.utils.register_keras_serializable(package="Similarity")
class ManhattanDistance(Distance):
    """Compute pairwise Manhattan distances between embeddings.

    The [Manhattan Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
    is the sum of the lengths of the projections of the line segment between
    two embeddings onto the Cartesian axes. The larger the distance the more
    dissimilar the embeddings are.
    """

    def __init__(self, name: str = "manhattan", **kwargs):
        "Init Manhattan distance"
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, query_embeddings: FloatTensor, key_embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch of embeddings.

        Args:
            query_embeddings: Embeddings to compute the pairwise one.
            key_embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
        q_rs = tf.reshape(query_embeddings, shape=[tf.shape(query_embeddings)[0], -1])
        k_rs = tf.reshape(key_embeddings, shape=[tf.shape(key_embeddings)[0], -1])
        deltas = tf.expand_dims(q_rs, axis=1) - tf.expand_dims(k_rs, axis=0)
        distances: FloatTensor = tf.norm(deltas, 1, axis=2)
        return distances
