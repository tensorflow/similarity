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
class SNRDistance(Distance):
    """
    Computes pairwise SNR distances between embeddings.

    The [Signal-to-Noise Ratio distance](https://arxiv.org/abs/1904.02616)
    is the ratio of noise variance to the feature variance.
    """

    def __init__(self, name: str = "snr", **kwargs):
        "Init SNR distance"
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, query_embeddings: FloatTensor, key_embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise snr distances for a given batch of embeddings.
        SNR(i, j): anchor i and compared feature j
        SNR(i,j) may not be equal to SNR(j, i)

        Args:
            query_embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """
        # Calculating feature variance for each example
        anchor_var = tf.math.reduce_variance(query_embeddings, axis=1)

        # Calculating pairwise noise variances
        q_rs = tf.reshape(query_embeddings, shape=[tf.shape(query_embeddings)[0], -1])
        k_rs = tf.reshape(key_embeddings, shape=[tf.shape(key_embeddings)[0], -1])
        delta = tf.expand_dims(q_rs, axis=1) - tf.expand_dims(k_rs, axis=0)
        noise_var = tf.math.reduce_variance(delta, axis=2)

        distances: FloatTensor = tf.divide(noise_var, tf.expand_dims(anchor_var, axis=1))

        return distances
