import tensorflow as tf
from tensorflow_similarity.algebra import masked_min, masked_max
from tensorflow_similarity.algebra import build_masks


def test_mask():
    batch_size = 16
    labels = tf.random.uniform((batch_size, 1), 0, 10, dtype=tf.int32)
    positive_mask, negative_mask = build_masks(labels, batch_size)
    assert positive_mask[0][0] == False
    assert positive_mask[5][5] == False

    combined = tf.math.logical_or(negative_mask, positive_mask)
    assert combined[0][0] == False
    for i in range(1, batch_size):
        assert combined[0][i] == True
        assert combined[i][0] == True


def test_masked_max():
    distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 2.0, 1.0, 0.0]],
                            dtype=tf.float32)
    mask = tf.constant([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=tf.float32)
    vals, arg_max = masked_max(distances, mask)

    assert vals.shape == (2, 1)
    assert arg_max.shape == (2,)
    assert vals[0] == [3.0]
    assert vals[1] == [2.0]
    assert arg_max[0] == [2]
    assert arg_max[1] == [1]


def test_arg_max_all_unmasked_vals_lt_zero():
    # Ensure reduce_max works when all unmasked vals < 0.0.
    distances = tf.constant([
             [-7.0, -2.0, 7.0, -9.0],
             [-7.0, 1e-05, 7.0, -9.0]],
            dtype=tf.float32)
    mask = tf.constant([[0, 0, 0, 1], [0, 1, 0, 0]], dtype=tf.float32)
    vals, arg_max = masked_max(distances, mask)

    assert vals.shape == (2, 1)
    assert arg_max.shape == (2,)
    assert vals[0] == [-9.0]
    assert vals[1] == [1e-05]
    assert arg_max[0] == [3]
    assert arg_max[1] == [1]


def test_masked_min():
    distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 1.0, 0.0]],
                            dtype=tf.float32)
    mask = tf.constant([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=tf.float32)
    vals, arg_min = masked_min(distances, mask)

    assert vals.shape == (2, 1)
    assert arg_min.shape == (2,)
    assert vals[0] == [2.0]
    assert vals[1] == [1.0]
    assert arg_min[0] == [1]
    assert arg_min[1] == [2]


def test_arg_min_all_unmasked_vals_gt_zero():
    # Ensure reduce_max works when all unmasked vals > 0.0.
    distances = tf.constant([
             [-7.0, -2.0, 7.0, -9.0],
             [-1e-06, -2.0, 7.0, -9.0]],
            dtype=tf.float32)
    mask = tf.constant([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=tf.float32)
    vals, arg_min = masked_min(distances, mask)

    assert vals.shape == (2, 1)
    assert arg_min.shape == (2,)
    assert vals[0] == [7.0]
    assert vals[1] == [-1e-06]
    assert arg_min[0] == [2]
    assert arg_min[1] == [0]
