import pytest
import tensorflow as tf
from tensorflow_similarity.distances import CosineDistance, InnerProductSimilarity
from tensorflow_similarity.distances import EuclideanDistance
from tensorflow_similarity.distances import ManhattanDistance
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


def test_inner_product_similarity():
    # pairwise
    a = tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    d = InnerProductSimilarity()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 12


def test_inner_product_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    d = InnerProductSimilarity()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 2


def test_inner_product_vals():
    a = tf.nn.l2_normalize([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]], axis=-1)
    d = InnerProductSimilarity()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert vals[0][0] == 1
    assert vals[0][1] == 0.68138516


def test_cosine_same():
    # pairwise
    a = tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    d = CosineDistance()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_cosine_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    d = CosineDistance()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
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
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 10


def test_euclidean_same():
    a = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
    d = EuclideanDistance()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_euclidean_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [0.0, -1.0]])
    d = EuclideanDistance()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
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
    assert tf.math.reduce_all(tf.shape(vals) == (4, 4))
    assert tf.reduce_all(tf.math.equal(vals, expected))


def test_manhattan_same():
    a = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
    d = ManhattanDistance()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_manhattan_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [0.0, -1.0]])
    d = ManhattanDistance()
    vals = d(a)
    assert tf.math.reduce_all(tf.shape(vals) == (2, 2))
    assert tf.round(tf.reduce_sum(vals)) == 4
