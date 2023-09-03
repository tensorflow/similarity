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

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor

from .distance import Distance


@tf.keras.utils.register_keras_serializable(package="Similarity")
class EuclideanDistance(Distance):
    """Compute pairwise euclidean distances between embeddings.

    The [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance)
    is the standard distance to measure the line segment between two embeddings
    in the Cartesian point. The larger the distance the more dissimilar
    the embeddings are.
    """

    def __init__(self, name: str = "euclidean", **kwargs):
        "Init Euclidean distance"
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
        q_squared_norm = tf.math.square(query_embeddings)
        q_squared_norm = tf.math.reduce_sum(q_squared_norm, axis=1, keepdims=True)

        k_squared_norm = tf.math.square(key_embeddings)
        k_squared_norm = tf.math.reduce_sum(k_squared_norm, axis=1, keepdims=True)

        distances: FloatTensor = 2.0 * tf.linalg.matmul(query_embeddings, key_embeddings, transpose_b=True)
        distances = q_squared_norm - distances + tf.transpose(k_squared_norm)

        # Avoid NaN and inf gradients when back propagating through the sqrt.
        # values smaller than 1e-18 produce inf for the gradient, and 0.0
        # produces NaN. All values smaller than 1e-13 should produce a gradient
        # of 1.0.
        dist_mask = tf.math.greater_equal(distances, 1e-18)
        distances = tf.math.maximum(distances, 1e-18)
        distances = tf.math.sqrt(distances) * tf.cast(dist_mask, tf.float32)

        return distances


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SquaredEuclideanDistance(Distance):
    """Compute pairwise squared Euclidean distance.

    The [Squared Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance) is
    a distance that varies from 0 (similar) to infinity (dissimilar).
    """

    def __init__(self, name: str = "squared_euclidean", **kwargs):
        "Init Squared Euclidean distance"
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
        q_squared_norm = tf.math.square(query_embeddings)
        q_squared_norm = tf.math.reduce_sum(q_squared_norm, axis=1, keepdims=True)

        k_squared_norm = tf.math.square(key_embeddings)
        k_squared_norm = tf.math.reduce_sum(k_squared_norm, axis=1, keepdims=True)

        distances: FloatTensor = 2.0 * tf.linalg.matmul(query_embeddings, key_embeddings, transpose_b=True)
        distances = q_squared_norm - distances + tf.transpose(k_squared_norm)
        distances = tf.math.maximum(distances, 0.0)

        return distances
