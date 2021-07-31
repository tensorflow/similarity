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
from typing import Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.types import FloatTensor, IntTensor
from .metric_loss import MetricLoss


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def circle_loss(labels: IntTensor,
                embeddings: FloatTensor,
                distance: Callable,
                gamma: float = 128,
                margin: float = 0.25):
    """Circle loss computations

    Args:
        labels: Labels associated with the embeddings

        embeddings: Embeddings as infered by the model.

        distance: Distance function to use to compute the pairwise.

        gamma: Scaling term. Defaults to 256. Good starting value
        according to figure 3. Should be hypertuned.

        margin: Distance minimal margin. Defaults to 0.25

    Returns:
        loss
    """

    # label
    batch_size = tf.size(labels)

    # [distances]
    pairwise_distances = distance(embeddings)

    # [masks] -> filter to keep only the relevant value - zero the rest
    positive_mask, negative_mask = build_masks(labels, batch_size)
    positive_mask = tf.cast(positive_mask, dtype='float32')
    negative_mask = tf.cast(negative_mask, dtype='float32')

    # [weights] from  (5) in 3.1 using optim values of 3.2
    # Implementation note: we do all the computation on the full pairwise and
    # filter at then end to keep only relevant values.

    # positive weights
    optim_pos = 1 + margin  # as in 3.2
    pos_weights = optim_pos - pairwise_distances  # (5) in 3.1
    pos_weights = tf.maximum(pos_weights, 0.0)  # clip at zero
    pos_weights = pos_weights * positive_mask  # filter

    # negative weights
    optim_neg = -1 * margin  # as in 3.2
    neg_weights = pairwise_distances - optim_neg  # (5) in 3.1
    neg_weights = tf.maximum(neg_weights, 0.0)  # clip at zero
    neg_weights = neg_weights * negative_mask  # filter

    # distances filtering
    # /2 because we have a pairwise so each distance is counted twice
    pos_dists = (pairwise_distances * positive_mask) / 2
    neg_dists = (pairwise_distances * negative_mask) / 2

    # applying weights as in (4) in 3.1
    pos_wdists = pos_weights * pos_dists
    neg_wdists = neg_weights * neg_dists

    # as in (4) in 3.1 reduce
    # ! paper is wrong they inverted positive and negatives
    loss = tf.math.subtract(pos_wdists, neg_wdists)

    # scale
    loss = gamma * loss

    # reduce
    loss = tf.reduce_logsumexp(loss)

    return loss

@tf.keras.utils.register_keras_serializable(package="Similarity")
class CircleLoss(MetricLoss):
    """Computes the CircleLoss

    Circle Loss: A Unified Perspective of Pair Similarity Optimization
    https://arxiv.org/abs/2002.10857


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
                 gamma: float = 256,
                 margin: float = 0.25,
                 name: str = None):
        """Initializes a CircleLoss

        Args:
            distance: Which distance function to use to compute
            the pairwise distances between embeddings. Defaults to 'cosine'.

            gamma: Scaling term. Defaults to 256. Good starting value
            according to figure 3. Should be hypertuned.

            margin: Margin term. Defaults to 0.25 as in the paper.

            name: Loss name. Defaults to None.

        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance

        super().__init__(circle_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         gamma=gamma,
                         margin=margin)
