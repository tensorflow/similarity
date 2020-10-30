import tensorflow as tf
from tensorflow_similarity.metrics import pairwise_cosine


def test_cosine_same():
    a = tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    vals = pairwise_cosine(a)
    assert tf.round(tf.reduce_sum(vals)) == 0


def test_cosine_opposite():
    a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
    vals = pairwise_cosine(a)
    assert tf.round(tf.reduce_sum(vals)) == 2
