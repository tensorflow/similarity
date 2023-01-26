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
from __future__ import annotations

from typing import Any

import tensorflow as tf

from tensorflow_similarity.algebra import masked_max, masked_min
from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor


def positive_distances(
    positive_mining_strategy: str,
    distances: FloatTensor,
    positive_mask: BoolTensor,
) -> tuple[FloatTensor, FloatTensor]:
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
    if positive_mining_strategy == "hard":
        positive_distances, pos_idxs = masked_max(distances, positive_mask)
    elif positive_mining_strategy == "easy":
        positive_distances, pos_idxs = masked_min(distances, positive_mask)
    else:
        raise ValueError("Invalid positive mining strategy")

    return positive_distances, pos_idxs


def negative_distances(
    negative_mining_strategy: str,
    distances: FloatTensor,
    negative_mask: BoolTensor,
    positive_mask: BoolTensor,
) -> tuple[FloatTensor, FloatTensor]:
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
    if negative_mining_strategy == "hard":
        # find the *non-zero* minimal distance between negative labels
        negative_distances, neg_idxs = masked_min(distances, negative_mask)
    elif negative_mining_strategy == "semi-hard":
        # Find the negative label with the minimal distance that is greater
        # than the maximal positive distance. If no such negative exists,
        # i.e., max(d(a,n)) < max(d(a,p)), then use the maximal negative
        # distance, as in the easy case.

        # find max value of positive distance
        max_positive, _ = masked_max(distances, positive_mask)

        # select distance that are above the max positive distance
        greater_distances = tf.math.greater(distances, max_positive)

        # combine greater_distances with negative mask: keep negative
        # value if greater, otherwise set to value from easy mask.
        _, max_neg_idxs = masked_max(distances, negative_mask)
        easy_mask = semi_hard_easy_mask(distances, max_neg_idxs)
        semi_hard_mask = tf.where(greater_distances, negative_mask, easy_mask)

        # find the  minimal distance between negative labels above threshold
        negative_distances, neg_idxs = masked_min(distances, semi_hard_mask)

    elif negative_mining_strategy == "easy":
        # find the maximal distance between negative labels
        negative_distances, neg_idxs = masked_max(distances, negative_mask)
    else:
        raise ValueError("Invalid negative mining strategy")

    return negative_distances, neg_idxs


def semi_hard_easy_mask(
    distances: FloatTensor,
    max_neg_idxs: IntTensor,
) -> BoolTensor:
    """Compute the fallback easy mask for semi-hard mining.

    Find the negative label with the maximal distance that is less than or
    equal to the maximal positive distance. This is used in the semi-hard
    mining for the case when no negative label is found that is greater
    than the maximal positive distance.
    """
    empty = tf.zeros_like(distances, dtype=tf.bool)
    updates = tf.ones(tf.shape(max_neg_idxs)[0], dtype=tf.bool)
    # tf.tensor_scatter_nd_update requires both the row and col idxs.
    # here we use a range for the row idxs and take the max_neg_idxs as the cols.
    row_idxs = tf.range(tf.shape(max_neg_idxs)[0])
    col_idxs = tf.cast(max_neg_idxs, dtype=row_idxs.dtype)
    indicies = tf.concat(
        (tf.expand_dims(row_idxs, axis=-1), tf.expand_dims(col_idxs, axis=-1)),
        axis=1,
    )

    # easy mask is a boolean tensor because we cast empty and updates to bool.
    easy_mask: BoolTensor = tf.tensor_scatter_nd_update(empty, indicies, updates)

    return easy_mask


def compute_loss(
    positive_distances: FloatTensor,
    negative_distances: FloatTensor,
    margin: float | None,
) -> Any:
    """Compute the final loss.

    Args:
        positive_distances: An [n,1] FloatTensor of positive distances.

        negative_distances: An [n,1] FloatTensor of negative distances.

        margin: [description]. Use soft margin if None otherwise use explicit margin.

    Returns:
        An [n,1] FloatTensor containing the loss for each example.
    """
    loss = tf.math.subtract(positive_distances, negative_distances)

    if margin is None:
        loss = logsumexp(loss, tf.ones_like(loss))
    else:
        loss = tf.math.add(loss, margin)
        loss = tf.maximum(loss, 0.0)  # numeric stability

    return loss


def logsumexp(pairwise_distances: FloatTensor, mask: FloatTensor) -> Any:
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
    raw_max = tf.math.reduce_max(pairwise_distances, axis=1, keepdims=True)

    my_max = tf.stop_gradient(tf.where(tf.math.is_finite(raw_max), raw_max, tf.zeros_like(raw_max)))

    x = tf.math.subtract(pairwise_distances, my_max)
    x = tf.math.exp(x)
    x = tf.math.multiply(x, mask)
    x = tf.math.reduce_sum(x, axis=1, keepdims=True)
    offset = tf.math.exp(-my_max)
    x = tf.math.log(offset + x)
    x = tf.math.add(x, my_max)

    return x
