import tensorflow as tf
import numpy as np
from tensorflow_similarity.metrics import pairwise_cosine, cosine


def angular_distance_np(feature):
    """Computes the angular distance matrix in numpy.
    Args:
      feature: 2-D numpy array of size [number of data, feature dimension]
    Returns:
      angular_distances: 2-D numpy array of size
        [number of data, number of data].
    """

    # l2-normalize all features
    normed = feature / np.linalg.norm(feature, ord=2, axis=1, keepdims=True)
    cosine_similarity = normed @ normed.T
    inverse_cos_sim = 1 - cosine_similarity

    return inverse_cos_sim


def test_cosine_same():
    # pairwise
    a = tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    vals = pairwise_cosine(a)
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_cosine_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    vals = pairwise_cosine(a)
    assert tf.round(tf.reduce_sum(vals)) == 2


def test_cosine_vals():
    a = tf.constant([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]])
    vals = pairwise_cosine(a)
    assert vals[0][0] == 0
    assert vals[0][1] == 0.31861484


def test_direct_cosine():
    a = tf.constant([[0.1, 0.3, 0.2]])
    b = tf.constant([[0.0, 0.1, 0.5]])
    assert cosine(a, b) == 0.31861484
