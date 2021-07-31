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
        positive_distances, pos_idxs = (
                masked_max(distances, positive_mask))
    elif positive_mining_strategy == 'easy':
        positive_distances, pos_idxs = (
                masked_min(distances, positive_mask))
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
        negative_mining_strategy (str, optional): [description].
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
        negative_distances, neg_idxs = (
                masked_min(distances, negative_mask))
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
        negative_distances, neg_idxs = (
                masked_min(distances, semi_hard_mask))

    elif negative_mining_strategy == 'easy':
        # find the maximal distance between negative labels
        negative_distances, neg_idxs = (
                masked_max(distances, negative_mask))
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
        soft_margin (bool, optional): [description]. Defaults to False.
        margin (float, optional): [description]. Defaults to 1.0.

    Returns:
        An [n,1] FloatTensor containing the loss for each example.
    """
    if soft_margin:
        loss = tf.math.subtract(positive_distances, negative_distances)
        loss = tf.math.exp(loss)
        loss = tf.math.log1p(loss)
    else:
        loss = tf.math.subtract(positive_distances, negative_distances)
        loss = tf.math.add(loss, margin)
        loss = tf.maximum(loss, 0.0)  # numeric stability

    return loss
