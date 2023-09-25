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
    from ..types import FloatTensor

from .distance import Distance


@tf.keras.utils.register_keras_serializable(package="Similarity")
class InnerProductSimilarity(Distance):
    """Compute the pairwise inner product between embeddings.

    The [Inner product](https://en.wikipedia.org/wiki/Inner_product_space) is
    a measure of similarity where the more similar vectors have the largest
    values.

    NOTE! This is not a distance and is likely not what you want to use with
    the built in losses. At the very least this will flip the sign on the
    margin in many of the losses. This is likely meant to be used with custom
    loss functions that expect a similarity instead of a distance.
    """

    def __init__(self, name: str = "inner_product", **kwargs):
        "Init Inner product similarity"
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, query_embeddings: FloatTensor, key_embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise similarities for a given batch of embeddings.

        Args:
            query_embeddings: Embeddings to compute the pairwise one.
            key_embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
        sims: FloatTensor = tf.linalg.matmul(query_embeddings, key_embeddings, transpose_b=True)
        return sims
