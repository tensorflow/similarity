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
class CosineDistance(Distance):
    """Compute pairwise cosine distances between embeddings.

    The [Cosine Distance](https://en.wikipedia.org/wiki/Cosine_similarity) is
    an angular distance that varies from 0 (similar) to 1 (dissimilar).
    """

    def __init__(self, name: str = "cosine", **kwargs):
        "Init Cosine distance"
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, query_embeddings: FloatTensor, key_embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch of embeddings.

        Args:
            query_embeddings: Embeddings to compute the pairwise one. The embeddings
            are expected to be normalized.
            key_embeddings: Embeddings to compute the pairwise one. The embeddings
            are expected to be normalized.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
        distances = 1 - tf.linalg.matmul(query_embeddings, key_embeddings, transpose_b=True)
        min_clip_distances: FloatTensor = tf.math.maximum(distances, 0.0)
        return min_clip_distances
