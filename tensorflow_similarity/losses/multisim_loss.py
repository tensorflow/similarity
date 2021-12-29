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
from tensorflow_similarity.algebra import build_masks, masked_max, masked_min
from tensorflow_similarity.types import FloatTensor, IntTensor
from .metric_loss import MetricLoss
from .utils import logsumexp


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def multisimilarity_loss(labels: IntTensor,
                         embeddings: FloatTensor,
                         distance: Callable,
                         alpha: float = 2.0,
                         beta: float = 40,
                         epsilon: float = 0.2,
                         lmda: float = 1.0) -> Any:
    """Multi Similarity loss computations

    Args:
        labels: labels associated with the embed.

        embeddings: Embedded examples.

        distance: Which distance function to use to compute the pairwise.

        alpha: The exponential weight for the positive pairs. Increasing alpha
        makes the logsumexp softmax closer to the max positive pair distance,
        while decreasing it makes it closer to max(P) + log(batch_size).

        beta: The exponential weight for the negative pairs. Increasing beta
        makes the logsumexp softmax closer to the max negative pair distance,
        while decreasing it makes the softmax closer to
        max(N) + log(batch_size).

        epsilon: Used to remove easy positive and negative pairs. We only keep
        positives that we greater than the (smallest negative pair - epsilon)
        and we only keep negatives that are less than the
        (largest positive pair + epsilon).

        lmda: Used to weight the distance. Below this distance, negatives are
        up weighted and positives are down weighted. Similarly, above this
        distance negatives are down weighted and positive are up weighted.

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

    # [pair mining using Similarity-P]
    # This is essentially hard mining the negative and positive pairs.

    # Keep all positives > Min(neg_dist - epsilon).
    neg_min, _ = masked_min(pairwise_distances, negative_mask)
    neg_min = tf.math.subtract(neg_min, epsilon)
    pos_sim_p_mask = tf.math.greater(pairwise_distances, neg_min)
    pos_sim_p_mask = tf.math.logical_and(pos_sim_p_mask, positive_mask)

    # Keep all negatives < Max(pos_dist + epsilon).
    pos_max, _ = masked_max(pairwise_distances, positive_mask)
    pos_max = tf.math.add(pos_max, epsilon)
    neg_sim_p_mask = tf.math.less(pairwise_distances, pos_max)
    neg_sim_p_mask = tf.math.logical_and(neg_sim_p_mask, negative_mask)

    # Mark all pairs where we have both valid negative and positive pairs.
    valid_anchors = tf.math.logical_and(
            tf.math.reduce_any(pos_sim_p_mask, axis=1),
            tf.math.reduce_any(neg_sim_p_mask, axis=1)
    )

    # Cast masks as floats to support multiply
    valid_anchors = tf.cast(valid_anchors, dtype='float32')
    pos_sim_p_mask_f32 = tf.cast(pos_sim_p_mask, dtype='float32')
    neg_sim_p_mask_f32 = tf.cast(neg_sim_p_mask, dtype='float32')

    # [Weight the remaining pairs using Similarity-S and Similarity-N]
    shifted_distances = pairwise_distances - lmda
    pos_dists = alpha * shifted_distances
    neg_dists = -1 * beta * shifted_distances

    # [compute loss]

    # Positive pairs with a distance above 0 will be up weighted.
    p_loss = logsumexp(pos_dists, pos_sim_p_mask_f32)
    p_loss = p_loss / alpha

    # Negative pairs with a distance below 0 will be up weighted.
    n_loss = logsumexp(neg_dists, neg_sim_p_mask_f32)
    n_loss = n_loss / beta

    # Remove any anchors that have empty neg or pos pairs.
    # NOTE: reshape is required here because valid_anchors is [m] and
    #       p_loss + n_loss is [m, 1].
    multisim_loss = tf.math.multiply(
            p_loss + n_loss,
            tf.reshape(valid_anchors, (-1, 1))
    )

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
                 alpha: float = 1.0,
                 beta: float = 20,
                 epsilon: float = 0.2,
                 lmda: float = 0.5,
                 name: str = 'MultiSimilarityLoss',
                 **kwargs):
        """Initializes the Multi Similarity Loss

        Args:
            distance: Which distance function to use to compute the pairwise
            distances between embeddings. Defaults to 'cosine'.

            alpha: The exponential weight for the positive pairs. Increasing
            alpha makes the logsumexp softmax closer to the max positive pair
            distance, while decreasing it makes it closer to
            max(P) + log(batch_size).

            beta: The exponential weight for the negative pairs. Increasing
            beta makes the logsumexp softmax closer to the max negative pair
            distance, while decreasing it makes the softmax closer to
            max(N) + log(batch_size).

            epsilon: Used to remove easy positive and negative pairs. We only
            keep positives that we greater than the (smallest negative pair -
            epsilon) and we only keep negatives that are less than the
            (largest positive pair + epsilon).

            lmda: Used to weight the distance. Below this distance, negatives
            are up weighted and positives are down weighted. Similarly, above
            this distance negatives are down weighted and positive are up
            weighted.

            name: Loss name. Defaults to MultiSimilarityLoss.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(multisimilarity_loss,
                         name=name,
                         distance=distance,
                         alpha=alpha,
                         beta=beta,
                         epsilon=epsilon,
                         lmda=lmda,
                         **kwargs)
