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
"""Utility functions for computing the metric loss."""

import tensorflow as tf
from tensorflow_similarity.algebra import masked_max, masked_min
from tensorflow_similarity.types import FloatTensor, BoolTensor
from typing import Any, Tuple


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def positive_distances(positive_mining_strategy: str,
                       distances: FloatTensor,
                       positive_mask: BoolTensor
                       ) -> Tuple[FloatTensor, FloatTensor]:
    """Positive distance computation.

    Args:
        positive_mining_strategy (str, optional): [description].
        distances: 2-D float `Tensor` of [n, n] pairwise distances.
        positive_mask: 2-D Boolean `Tensor` of [n, n] valid distance size.

    Raises:
        ValueError: Invalid positive mining strategy

    Returns:
      A Tuple of Tensors containing the positive distance values
      and the index for each example.
    """
    if positive_mining_strategy == 'hard':
        positive_distances, pos_idxs = (masked_max(distances, positive_mask))
    elif positive_mining_strategy == 'easy':
        positive_distances, pos_idxs = (masked_min(distances, positive_mask))
    else:
        raise ValueError('Invalid positive mining strategy')

    return positive_distances, pos_idxs


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def negative_distances(negative_mining_strategy: str,
                       distances: FloatTensor,
                       negative_mask: BoolTensor,
                       positive_mask: BoolTensor,
                       batch_size: int) -> Tuple[FloatTensor, FloatTensor]:
    """Negative distance computation.

    Args:
        negative_mining_strategy: What mining strategy to use for select the
        embedding from the different class.
        Available: {'hard', 'semi-hard', 'easy'}

        distances: 2-D float `Tensor` of [n, n] pairwise distances.

        negative_mask: 2-D Boolean `Tensor` of [n, n] valid distance size.

        positive_mask: 2-D Boolean `Tensor` of [n, n] valid distance size.

        batch_size: The current batch size.

    Raises:
        ValueError: Invalid negative mining strategy

    Returns:
      A Tuple of Tensors containing the negative distance values
      and the index for each example.
    """
    if negative_mining_strategy == 'hard':
        # find the *non-zero* minimal distance between negative labels
        negative_distances, neg_idxs = (masked_min(distances, negative_mask))
    elif negative_mining_strategy == 'semi-hard':
        # find the minimal distance between negative label gt than max distance
        # between positive labels
        # find max value of positive distance
        max_positive, _ = masked_max(distances, positive_mask)

        # select distance that are above the max positive distance
        greater_distances = tf.math.greater(distances, max_positive)

        # combine with negative mask: keep negative value if greater,
        # zero otherwise
        empty = tf.zeros((batch_size, batch_size), dtype=tf.bool)
        semi_hard_mask = tf.where(greater_distances, negative_mask, empty)

        # find the  minimal distance between negative labels above threshold
        negative_distances, neg_idxs = (masked_min(distances, semi_hard_mask))

    elif negative_mining_strategy == 'easy':
        # find the maximal distance between negative labels
        negative_distances, neg_idxs = (masked_max(distances, negative_mask))
    else:
        raise ValueError('Invalid negative mining strategy')

    return negative_distances, neg_idxs


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def compute_loss(positive_distances: FloatTensor,
                 negative_distances: FloatTensor,
                 soft_margin: bool,
                 margin: float) -> Any:
    """Compute the final loss.

    Args:
        positive_distances: An [n,1] FloatTensor of positive distances.

        negative_distances: An [n,1] FloatTensor of negative distances.

        soft_margin: [description]. Defaults to False.

        margin: [description]. Defaults to 1.0.

    Returns:
        An [n,1] FloatTensor containing the loss for each example.
    """

    loss = tf.math.subtract(positive_distances, negative_distances)

    if soft_margin:
        loss = tf.reduce_logsumexp(loss)
    else:
        loss = tf.math.add(loss, margin)
        loss = tf.maximum(loss, 0.0)  # numeric stability

    return loss


@tf.keras.utils.register_keras_serializable(package="Similarity")
@tf.function
def logsumexp(pairwise_distances: FloatTensor,
              mask: FloatTensor) -> Any:
    """Compute the LogSumExp across axis 1 of the pairwise distance matrix.

    This function:
    * Avoids numerical instablity when the inputs are large or small.
    * Adds 1 to the reduce_sum of the exp to ensure the loss is positive
    * masks out the result of exp to ensure that masked values are not included
      in the log sum.

    Args:
        pairwise_distance: A 2D FloatTensor of the pairwise distances.

        mask: A 2D FloatTensor with 1.0 for all valid values and 0.0
        everywhere else.

    returns:
        A [n, 1] FloatTensor containing the per example LogSumExp values.
    """
    raw_max = tf.math.reduce_max(
        pairwise_distances,
        axis=1,
        keepdims=True
    )
    my_max = tf.stop_gradient(
        tf.where(
            tf.math.is_finite(raw_max),
            raw_max,
            tf.zeros_like(raw_max)
        )
    )
    x = tf.math.subtract(pairwise_distances, my_max)
    x = tf.math.exp(x) * mask
    x = tf.math.reduce_sum(x, axis=1, keepdims=True)
    x = tf.math.log(1 + x)
    x = tf.math.add(x, my_max)

    return x
