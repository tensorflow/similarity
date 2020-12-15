import tensorflow as tf
from tensorflow_similarity.algebra import masked_minimum, masked_maximum
from tensorflow_similarity.algebra import build_masks


def test_mask():
    batch_size = 16
    labels = tf.random.uniform((batch_size, 1), 0, 10, dtype=tf.int32)
    positive_mask, negative_mask = build_masks(labels, batch_size)
    assert positive_mask[0][0] == 0
    assert positive_mask[5][5] == 0

    combined = negative_mask + positive_mask
    assert combined[0][0] == 0
    for i in range(1, batch_size):
        assert combined[0][i] == 1
        assert combined[i][0] == 1


def test_masked_maximum():
    distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 2.0, 1.0, 0.0]],
                            dtype=tf.float32)
    mask = tf.constant([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=tf.float32)
    vals = masked_maximum(distances, mask)
    assert vals.shape == (2, 1)
    assert vals[0] == [3.0]
    assert vals[1] == [2.0]


def test_masked_minimum():
    distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 1.0, 0.0]],
                            dtype=tf.float32)
    mask = tf.constant([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=tf.float32)
    vals = masked_minimum(distances, mask)
    print(vals)
    assert vals.shape == (2, 1)
    assert vals[0] == [2.0]
    assert vals[1] == [1.0]
