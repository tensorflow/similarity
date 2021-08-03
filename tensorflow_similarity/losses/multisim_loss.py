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
"""Multi Similarity Loss"""

import tensorflow as tf
from typing import Any, Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.types import FloatTensor, IntTensor
from .metric_loss import MetricLoss


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def multisimilarity_loss(labels: IntTensor,
                         embeddings: FloatTensor,
                         distance: Callable,
                         alpha: float = 2.0,
                         beta: float = 50,
                         base: float = 0.0) -> Any:
    """Multi Similarity loss computations

    Args:
        labels: labels associated with the embed

        embeddings: Embedded examples.

        distance: Which distance function to use to compute the pairwise

        alpha: The exponential weight for the positive pairs.

        beta: The exponential weight for the negative pairs.

        base: The shift in exponent for both pos and neg pairs.

    Returns:
        Loss: The loss value for the current batch.
    """

    # [Label]
    # ! Weirdness to be investigated
    # do not remove this code. It is actually needed for specific situation
    # Reshape label tensor to [batch_size, 1] if not already in that format.
    # labels = tf.reshape(labels, (labels.shape[0], 1))
    batch_size = tf.size(labels)

    # [distances]
    pairwise_distances = distance(embeddings)

    # [masks]
    positive_mask, negative_mask = build_masks(labels, batch_size)
    positive_mask_f32 = tf.cast(positive_mask, dtype='float32')
    negative_mask_f32 = tf.cast(negative_mask, dtype='float32')

    padding = tf.constant([[0, 0], [0, 1]])

    # keep_mask=pos_mask.bool()
    pos_exp = pairwise_distances - base
    pos_exp = tf.math.multiply(pos_exp, positive_mask_f32)
    pos_exp_zero = tf.math.multiply(
            tf.fill(tf.shape(pos_exp), -1e6),
            tf.cast(tf.math.logical_not(positive_mask), dtype='float32')
    )
    # add_one=True
    pos_exp = tf.pad(pos_exp + pos_exp_zero, padding, "CONSTANT")

    # keep_mask=neg_mask.bool()
    neg_exp = base - pairwise_distances
    neg_exp = tf.math.multiply(neg_exp, negative_mask_f32)
    neg_exp_zero = tf.math.multiply(
            tf.fill(tf.shape(neg_exp), -1e6),
            tf.cast(tf.math.logical_not(negative_mask), dtype='float32')
    )
    # add_one=True
    neg_exp = tf.pad(neg_exp + neg_exp_zero, padding, "CONSTANT")

    # [compute loss]
    pos_loss = tf.math.reduce_logsumexp(alpha * pos_exp, axis=1, keepdims=True)
    pos_loss = tf.math.multiply(
            pos_loss,
            tf.cast(
                tf.math.reduce_any(positive_mask, axis=1, keepdims=True),
                dtype='float32'
            )
    )
    pos_loss *= (1.0/alpha)

    neg_loss = tf.math.reduce_logsumexp(beta * neg_exp, axis=1, keepdims=True)
    neg_loss = tf.math.multiply(
            neg_loss,
            tf.cast(
                tf.math.reduce_any(negative_mask, axis=1, keepdims=True),
                dtype='float32'
            )
    )
    neg_loss *= (1.0/beta)

    multisim_loss = pos_loss + neg_loss

    return multisim_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class MultiSimilarityLoss(MetricLoss):
    """Computes the multi similarity loss in an online fashion.


    `y_true` must be  a 1-D integer `Tensor` of shape (batch_size,).
    It's values represent the classes associated with the examples as
    **integer  values**.

    `y_pred` must be 2-D float `Tensor`  of L2 normalized embedding vectors.
    you can use the layer `tensorflow_similarity.layers.L2Embedding()` as the
    last layer of your model to ensure your model output is properly
    normalized.
    """

    def __init__(self,
                 distance: Union[Distance, str] = 'cosine',
                 alpha: float = 2.0,
                 beta: float = 50,
                 base: float = 0.0,
                 name: str = None):
        """Initializes the Multi Similarity Loss

        Args:
            distance: Which distance function to use to compute the pairwise
            distances between embeddings. Defaults to 'cosine'.

            alpha: The exponential weight for the positive pairs.

            beta: The exponential weight for the negative pairs.

            base: The shift in exponent for both pos and neg pairs.

            name: Loss name. Defaults to None.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance
        # sanity checks

        super().__init__(multisimilarity_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         alpha=alpha,
                         beta=beta,
                         base=base)
