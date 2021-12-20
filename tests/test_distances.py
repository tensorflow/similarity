import pytest
import tensorflow as tf
import numpy as np
from tensorflow_similarity.distances import CosineDistance, InnerProductSimilarity
from tensorflow_similarity.distances import EuclideanDistance
from tensorflow_similarity.distances import ManhattanDistance
from tensorflow_similarity.distances import SNRDistance
from tensorflow_similarity.distances import distance_canonicalizer
from tensorflow_similarity.distances import DISTANCES


def test_distance_mapping():
    for d in DISTANCES:

        # self naming
        d2 = distance_canonicalizer(d.name)
        assert d2.name == d.name

        # aliases
        for a in d.aliases:
            d2 = distance_canonicalizer(a)
            assert d2.name == d.name


def test_distance_passthrough():
    "Canonilizer is expected to return distance object as is"
    d = EuclideanDistance()
    d2 = distance_canonicalizer(d)
    assert d == d2


def test_non_existing_distance():
    with pytest.raises(ValueError):
        distance_canonicalizer('notadistance')


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
    d = CosineDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_cosine_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    d = CosineDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 2


def test_cosine_vals():
    a = tf.nn.l2_normalize([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]], axis=-1)
    d = CosineDistance()
    vals = d(a)
    assert vals[0][0] == 0
    assert vals[0][1] == 0.31861484


def test_euclidean():
    a = tf.convert_to_tensor([[0.0, 3.0], [4.0, 0.0]])
    d = EuclideanDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 10


def test_euclidean_same():
    a = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
    d = EuclideanDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_euclidean_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [0.0, -1.0]])
    d = EuclideanDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 4


def test_manhattan():
    a = tf.convert_to_tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [3.0, 0.0]
    ])
    d = ManhattanDistance()
    vals = d(a)
    expected = tf.constant([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 4.0],
        [2.0, 1.0, 0.0, 3.0],
        [3.0, 4.0, 3.0, 0.0]
    ])
    assert tf.reduce_all(tf.math.equal(vals, expected))


def test_manhattan_same():
    a = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
    d = ManhattanDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_manhattan_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [0.0, -1.0]])
    d = ManhattanDistance()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 4


def test_innerprod():
    a = [[1, 2, 3], [1, 3, 3]]
    d = InnerProductSimilarity()
    vals = d(a)
    assert tf.round(tf.reduce_sum(vals)) == 65


def test_snr_dist():
    """
    Comparing SNRDistance with simple loop based implementation
    of SNR distance.
    """
    num_inputs = 3
    dims = 5
    x = np.random.uniform(0, 1, (num_inputs, dims))

    # Computing SNR distance values using loop
    snr_pairs = []
    for i in range(num_inputs):
        row = []
        for j in range(num_inputs):
            dist = np.var(x[i]-x[j])/np.var(x[i])
            row.append(dist)
        snr_pairs.append(row)
    snr_pairs = np.array(snr_pairs)

    x = tf.convert_to_tensor(x)
    snr_distances = SNRDistance()(x).numpy()
    assert np.all(snr_distances >= 0)
    diff = snr_distances - snr_pairs
    assert np.all(np.abs(diff) < 1e-4)
