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
"""Circle Loss
    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857
"""

import tensorflow as tf
from typing import Any, Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.types import FloatTensor, IntTensor
from .metric_loss import MetricLoss
from .utils import logsumexp


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def circle_loss(query_labels: IntTensor,
                query_embeddings: FloatTensor,
                key_labels: IntTensor,
                key_embeddings: FloatTensor,
                distance: Callable,
                remove_diagonal: bool = True,
                gamma: float = 80,
                margin: float = 0.4) -> Any:
    """Circle loss computations

    The original paper used cosine similarity while this loss has been modified
    to work with cosine distance.

    Args:
        query_labels: labels associated with the query embed.

        query_embeddings: Embedded query examples.

        key_labels: labels associated with the key embed.

        key_embeddings: Embedded key examples.

        distance: Which distance function to use to compute the pairwise
        distances between embeddings. The distance is expected to be
        between [0, 2]. Defaults to 'cosine'.

        remove_diagonal: Bool. If True, will set diagonal to False in positive pair mask

        gamma: Scaling term. Defaults to 80. Note: Large values cause the
        LogSumExp to return the Max pair and reduces the weighted mixing of all
        pairs. Should be hypertuned.

        margin: Used to weight the distance. Below this distance, negatives are
        up weighted and positives are down weighted. Similarly, above this
        distance negatives are down weighted and positive are up weighted.
        Defaults to 0.4.

    Returns:
        Loss: The loss value for the current batch.
    """

    # Switched from what's in the paper to work with distance instead of
    # similarity.
    optim_pos = margin
    optim_neg = 1 + margin
    delta_pos = margin
    delta_neg = 1 - margin

    # label
    batch_size = tf.size(query_labels)

    # [distances]
    pairwise_distances = distance(query_embeddings, key_embeddings)

    # [masks] -> filter to keep only the relevant value - zero the rest
    positive_mask, negative_mask = build_masks(
        query_labels,
        key_labels,
        batch_size=batch_size,
        remove_diagonal=remove_diagonal,
    )
    valid_anchors = tf.math.logical_and(
            tf.math.reduce_any(positive_mask, axis=1),
            tf.math.reduce_any(negative_mask, axis=1)
    )

    # Cast masks as floats to support multiply
    valid_anchors = tf.cast(valid_anchors, dtype='float32')
    positive_mask = tf.cast(positive_mask, dtype='float32')
    negative_mask = tf.cast(negative_mask, dtype='float32')

    # [weights] from  (5) in 3.1 using optim values of 3.2
    # Implementation note: we do all the computation on the full pairwise and
    # filter at then end to keep only relevant values.

    # positive weights
    pos_weights = optim_pos + pairwise_distances  # (5) in 3.1
    pos_weights = pos_weights * positive_mask  # filter
    pos_weights = tf.maximum(pos_weights, 0.0)  # clip at zero

    # negative weights
    neg_weights = optim_neg - pairwise_distances  # (5) in 3.1
    neg_weights = neg_weights * negative_mask  # filter
    neg_weights = tf.maximum(neg_weights, 0.0)  # clip at zero

    # Subtract the between and within class margins
    pos_dists = delta_pos - pairwise_distances
    neg_dists = delta_neg - pairwise_distances

    # distances filtering
    # /2 because we have a pairwise so each distance is counted twice
    # applying weights as in (4) in 3.1
    pos_wdists = (-1 * gamma * pos_weights * pos_dists)  # / 2
    neg_wdists = (gamma * neg_weights * neg_dists)  # / 2

    p_loss = logsumexp(pos_wdists, positive_mask)
    n_loss = logsumexp(neg_wdists, negative_mask)

    # Remove any anchors that have empty neg or pos pairs.
    # NOTE: reshape is required here because valid_anchors is [m] and
    #       p_loss + n_loss is [m, 1].
    circle_loss = tf.math.multiply(
            p_loss + n_loss,
            tf.reshape(valid_anchors, (-1, 1))
    )

    return circle_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class CircleLoss(MetricLoss):
    """Computes the CircleLoss

    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857

    The original paper used cosine similarity while this loss has been
    modified to work with cosine distance.

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
                 gamma: float = 80.0,
                 margin: float = 0.40,
                 name: str = 'CircleLoss',
                 **kwargs):
        """Initializes a CircleLoss

        Args:
            distance: Which distance function to use to compute the pairwise
            distances between embeddings. The distance is expected to be
            between [0, 2]. Defaults to 'cosine'.

            gamma: Scaling term. Defaults to 80. Note: Large values cause the
            LogSumExp to return the Max pair and reduces the weighted mixing
            of all pairs. Should be hypertuned.

            margin: Used to weight the distance. Below this distance, negatives
            are up weighted and positives are down weighted. Similarly, above
            this distance negatives are down weighted and positive are up
            weighted. Defaults to 0.4.

            name: Loss name. Defaults to CircleLoss.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(circle_loss,
                         name=name,
                         distance=distance,
                         gamma=gamma,
                         margin=margin,
                         **kwargs)
