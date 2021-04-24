"Set of useful algebric functions used through the package"
from typing import Tuple
import tensorflow as tf
from .types import FloatTensor


def masked_maximum(distances: FloatTensor,
                   mask: FloatTensor,
                   dim: int = 1) -> Tuple[FloatTensor, FloatTensor]:
    """Computes the maximum values over masked pairwise distances.

    We need to use this formula to make sure all values are >=0.

    Args:
      distances (Tensor): 2-D float `Tensor` of [n, n] pairwise distances
      mask (Tensor): 2-D Boolean `Tensor` of [n, n] valid distance size.
      dim: The dimension over which to compute the maximum.

    Returns:
      Tensor: The maximum distance value
    """
    # Convert to dbl to avoid precision error in offset
    distances = tf.cast(distances, dtype=tf.float64)
    mask = tf.cast(mask, dtype=tf.float64)

    axis_min = tf.math.reduce_min(distances, dim, keepdims=True) - 1e-6
    masked_max = tf.math.multiply(distances - axis_min, mask)
    arg_max = tf.math.argmax(masked_max, dim)
    masked_max = tf.math.reduce_max(masked_max, dim, keepdims=True) + axis_min
    return tf.cast(masked_max, dtype=tf.float32), arg_max


def masked_minimum(distances: FloatTensor,
                   mask: FloatTensor,
                   dim: int = 1) -> Tuple[FloatTensor, FloatTensor]:
    """Computes the mimimal values over masked pairwise distances.

    Args:
      distances (Tensor): 2-D float `Tensor` of [n, n] pairwise distances
      mask (Tensor): 2-D Boolean `Tensor` of [n, n] valid distance size.
      dim: The dimension over which to compute the maximum.

    Returns:
      Tensor: The minimal distance value
    """
    # Convert to dbl to avoid precision error in offset
    distances = tf.cast(distances, dtype=tf.float64)
    mask = tf.cast(mask, dtype=tf.float64)

    axis_max = tf.math.reduce_max(distances, dim, keepdims=True) + 1e-6
    masked_min = tf.math.multiply(distances - axis_max, mask)
    arg_min = tf.math.argmin(masked_min, dim)
    masked_min = tf.math.reduce_min(masked_min, dim, keepdims=True) + axis_max
    return tf.cast(masked_min, dtype=tf.float32), arg_min


def build_masks(labels: FloatTensor,
                batch_size: int) -> Tuple[FloatTensor, FloatTensor]:
    """Build masks that allows to select only the positive or negatives
    embeddings.

    Args:
        labels (Tensor): 1D int `Tensor` that contains the class ids.
        batch_size (int): size of the batch.

    Returns:
        list: positive_mask, negative_mask
    """
    if tf.rank(labels) == 1:
        labels = tf.reshape(labels, (-1, 1))

    # same class mask
    positive_mask = tf.math.equal(labels, tf.transpose(labels))
    # not the same class
    negative_mask = tf.math.logical_not(positive_mask)

    # masks are treated as float32 moving forward
    positive_mask = tf.cast(positive_mask, tf.float32)
    negative_mask = tf.cast(negative_mask, tf.float32)

    # we need to remove the diagonal from postivie mask
    diag = tf.linalg.diag(tf.ones(batch_size))
    positive_mask = positive_mask - tf.cast(diag, tf.float32)
    return positive_mask, negative_mask
