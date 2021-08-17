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
"""Lifted Structured Loss
    Deep Metric Learning via Lifted Structured Feature Embedding.
    https://arxiv.org/abs/1511.06452
"""

import tensorflow as tf
from typing import Any, Callable, Union

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.algebra import build_masks
from tensorflow_similarity.types import FloatTensor, IntTensor
from .utils import negative_distances
from .metric_loss import MetricLoss


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def lifted_struct_loss(labels: IntTensor,
                       embeddings: FloatTensor,
                       distance: Callable,
                       positive_mining_strategy: str = 'hard',
                       negative_mining_strategy: str = 'easy',
                       soft_margin: bool = False,
                       margin: float = 1.0) -> Any:
    """Lifted Struct loss computations

    Args:
        labels: labels associated with the embed

        embeddings: Embedded examples.

        distance: Which distance function to use to compute the pairwise
        distances between embeddings. Defaults to 'cosine'.

        positive_mining_strategy: What mining strategy to use to select
        embedding from the same class. Defaults to 'hard'.
        Available: {'easy', 'hard'}

        negative_mining_strategy: What mining strategy to use for select the
        embedding from the different class. Defaults to 'easy'.
        Available: {'hard', 'semi-hard', 'easy'}

        soft_margin: [description]. Defaults to True. Use a soft margin
        instead of an explicit one.

        margin: Use an explicit value for the margin term. Defaults to 1.0.

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
    diff = margin - pairwise_distances

    # [masks]
    positive_mask, negative_mask = build_masks(labels, batch_size)
    positive_mask = tf.cast(positive_mask, dtype='float32')
    negative_mask = tf.cast(negative_mask, dtype='float32')

    # [Negative distances computation]
    neg_distances, _ = negative_distances(
            negative_mining_strategy,
            diff,
            negative_mask,
            positive_mask,
            batch_size,
    )

    max_elements = tf.math.maximum(
            neg_distances, tf.transpose(neg_distances)
    )
    diff_tiled = tf.tile(diff, [batch_size, 1])
    neg_mask_tiled = tf.tile(negative_mask, [batch_size, 1])
    max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

    loss_exp_left = tf.math.multiply(
        tf.math.exp(diff_tiled - max_elements_vect),
        neg_mask_tiled
    )
    loss_exp_left = tf.reshape(
        tf.math.reduce_sum(loss_exp_left, 1, keepdims=True,),
        [batch_size, batch_size],
    )

    loss_mat = loss_exp_left + tf.transpose(loss_exp_left)
    loss_mat = max_elements + tf.math.log(loss_mat)
    # Add the positive distance.
    loss_mat += pairwise_distances

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = tf.math.reduce_sum(positive_mask) / 2.0

    lifted_loss = tf.math.truediv(
        0.25
        * tf.math.reduce_sum(
            tf.math.square(
                tf.math.maximum(tf.math.multiply(loss_mat, positive_mask), 0.0)
            )
        ),
        num_positives,
    )

    return lifted_loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
class LiftedStructLoss(MetricLoss):
    """Computes the lifted struct loss in an online fashion.

    This loss encourages the positive distances between a pair of embeddings
    with the same labels to be smaller than the minimum negative distances
    between pair of embeddings of different labels.
    TDOD: Explain how this is different than standard triplet loss.
    See: https://arxiv.org/abs/1511.06452 for details.


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
                 positive_mining_strategy: str = 'hard',
                 negative_mining_strategy: str = 'easy',
                 soft_margin: bool = False,
                 margin: float = 1.0,
                 name: str = None):
        """Initializes the LiftedStuctLoss

        Args:
            distance: Which distance function to use to compute
            the pairwise distances between embeddings. Defaults to 'cosine'.

            positive_mining_strategy: What mining strategy to
            use to select embedding from the same class. Defaults to 'hard'.
            available: {'easy', 'hard'}

            negative_mining_strategy: What mining strategy to
            use for select the embedding from the different class.
            Defaults to 'easy'. Available: {'hard', 'semi-hard', 'easy'}

            soft_margi: [description]. Defaults to True.
            Use a soft margin instead of an explicit one.

            margin: Use an explicit value for the margin
            term. Defaults to 1.0.

            name: Loss name. Defaults to None.

        Raises:
            ValueError: Invalid positive mining strategy.
            ValueError: Invalid negative mining strategy.
            ValueError: Margin value is not used when soft_margin is set
                        to True.
        """

        # distance canonicalization
        distance = distance_canonicalizer(distance)
        self.distance = distance
        # sanity checks

        if positive_mining_strategy not in ['easy', 'hard']:
            raise ValueError('Invalid positive mining strategy')

        if negative_mining_strategy not in ['easy', 'hard', 'semi-hard']:
            raise ValueError('Invalid negative mining strategy')

        # Ensure users knows its one or the other
        if margin != 1.0 and soft_margin:
            raise ValueError('Margin value is not used when soft_margin is\
                              set to True')

        super().__init__(lifted_struct_loss,
                         name=name,
                         reduction=tf.keras.losses.Reduction.NONE,
                         distance=distance,
                         positive_mining_strategy=positive_mining_strategy,
                         negative_mining_strategy=negative_mining_strategy,
                         soft_margin=soft_margin,
                         margin=margin)
