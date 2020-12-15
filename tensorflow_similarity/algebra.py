"Set of useful algebric functions used through the package"
import tensorflow as tf


def masked_maximum(distances, mask, dim=1):
    """Computes the maximum values over masked pairwise distances.

    We need to use this formula to make sure all values are >=0.

    Args:
      distances (Tensor): 2-D float `Tensor` of [n, n] pairwise distances
      mask (Tensor): 2-D Boolean `Tensor` of [n, n] valid distance size.
      dim: The dimension over which to compute the maximum.

    Returns:
      Tensor: The maximum distance value
    """
    axis_min = tf.math.reduce_min(distances, dim, keepdims=True)
    masked_max = tf.math.multiply(distances - axis_min, mask)
    masked_max = tf.math.reduce_max(masked_max, dim, keepdims=True) + axis_min
    return masked_max


def masked_minimum(data, mask, dim=1):
    """Computes the mimimal values over masked pairwise distances.

    Args:
      distances (Tensor): 2-D float `Tensor` of [n, n] pairwise distances
      mask (Tensor): 2-D Boolean `Tensor` of [n, n] valid distance size.
      dim: The dimension over which to compute the maximum.

    Returns:
      Tensor: The minimal distance value
    """
    axis_max = tf.math.reduce_max(data, dim, keepdims=True)
    masked_min = tf.math.multiply(data - axis_max, mask)
    masked_min = tf.math.reduce_min(masked_min, dim, keepdims=True)
    masked_min = masked_min + axis_max
    return masked_min


def build_masks(labels, batch_size):
    """Build masks that allows to select only the positive or negatives
    embeddings.

    Args:
        labels (Tensor): 1D int `Tensor` that contains the class ids.
        batch_size (int): size of the batch.

    Returns:
        list: positive_mask, negative_mask
    """
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
