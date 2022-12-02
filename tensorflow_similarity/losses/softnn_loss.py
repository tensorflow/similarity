# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Soft Nearest Neighbors Loss.
    FaceNet: A Unified Embedding for Face Recognition and Clustering
    https://arxiv.org/abs/1902.01889
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf

from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor, IntTensor

from .metric_loss import MetricLoss


def soft_nn_loss(
    query_labels: IntTensor,
    query_embeddings: FloatTensor,
    key_labels: IntTensor,
    key_embeddings: FloatTensor,
    distance: Distance,
    temperature: float,
    remove_diagonal: bool = True,
) -> Any:
    """Computes the soft nearest neighbors loss.

    Args:
        query_labels: labels associated with the query embed.
        query_embeddings: Embedded query examples.
        key_labels: labels associated with the key embed.
        key_embeddings: Embedded key examples.
        distance: Which distance function to use to compute the pairwise.
        temperature: Controls relative importance given
                        to the pair of points.
        remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask

    Returns:
        loss: loss value for the current batch.
    """

    batch_size = tf.size(query_labels)
    eps = 1e-9

    pairwise_dist = distance(query_embeddings, key_embeddings)
    pairwise_dist = pairwise_dist / temperature
    negexpd = tf.math.exp(-pairwise_dist)

    # Mask out diagonal entries
    diag = tf.linalg.diag(tf.ones(batch_size, dtype=tf.bool))
    diag_mask = tf.cast(tf.logical_not(diag), dtype=tf.float32)
    negexpd = tf.math.multiply(negexpd, diag_mask)

    # creating mask to sample same class neighboorhood
    pos_mask, _ = build_masks(
        query_labels,
        key_labels,
        batch_size=batch_size,
        remove_diagonal=remove_diagonal,
    )
    pos_mask = tf.cast(pos_mask, dtype=tf.float32)

    # all class neighborhood
    alcn = tf.reduce_sum(negexpd, axis=1)

    # same class neighborhood
    sacn = tf.reduce_sum(tf.math.multiply(negexpd, pos_mask), axis=1)

    # exclude examples with unique class from loss calculation
    excl = tf.math.not_equal(tf.reduce_sum(pos_mask, axis=1), tf.zeros(batch_size))
    excl = tf.cast(excl, tf.float32)

    loss = tf.math.divide(sacn, alcn)
    loss = -tf.multiply(tf.math.log(eps + loss), excl)

    return loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SoftNearestNeighborLoss(MetricLoss):
    """Computes the soft nearest neighbors loss in an online fashion.

    Similar to TripletLoss, this loss compares intra- and inter-class
    distances. However, unlike TripletLoss, this loss uses all
    positive and negative examples in the batch.
    See: https://arxiv.org/abs/1902.01889 for the original paper.

    `labels` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `embeddings` must be 2-D float `Tensor` of embedding vectors.
    """

    def __init__(
        self,
        distance: Distance | str = "sql2",
        temperature: float = 1,
        name: str = "SoftNearestNeighborLoss",
        **kwargs,
    ):
        """Initializes the SoftNearestNeighborLoss Loss

        Args:
            `distance`: Which distance function to use to compute
                        the pairwise distances between embeddings.
                        Defaults to 'sql2'.

            `temperature`: Alters the value of loss function.
                           Defaults to 1.

            `name`: Loss name. Defaults to SoftNearestNeighborLoss.
        """
        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance
        self.temperature = temperature

        super().__init__(fn=soft_nn_loss, name=name, distance=distance, temperature=temperature, **kwargs)
