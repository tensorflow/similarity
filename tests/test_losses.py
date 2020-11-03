import tensorflow as tf
from tensorflow_similarity.losses import TripletLoss
from tensorflow_similarity.losses import _masked_maximum, _masked_minimum
from tensorflow_similarity.losses import _build_masks


def test_masked_maximum():
    distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 2.0, 1.0, 0.0]], dtype=tf.float32)
    mask = tf.constant([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=tf.float32)
    vals = _masked_maximum(distances, mask)
    assert vals.shape == (2, 1)
    assert vals[0] == [3.0]
    assert vals[1] == [2.0]


def test_masked_minimum():
    distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
    mask = tf.constant([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=tf.float32)
    vals = _masked_minimum(distances, mask)
    print(vals)
    assert vals.shape == (2, 1)
    assert vals[0] == [2.0]
    assert vals[1] == [1.0]


def test_mask():
    batch_size = 16
    labels = tf.random.uniform((batch_size, 1), 0, 10, dtype=tf.int32)
    positive_mask, negative_mask = _build_masks(labels, batch_size)
    assert positive_mask[0][0] == 0
    assert positive_mask[5][5] == 0

    combined = negative_mask + positive_mask
    assert combined[0][0] == 0
    for i in range(1, batch_size):
        assert combined[0][i] == 1
        assert combined[i][0] == 1


def test_triplet_loss():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs,), 0, 10, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 20), 0, 1)
    tpl = TripletLoss()
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss > 1


def test_triplet_loss_easy():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs,), 0, 3, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 16), 0, 1)
    tpl = TripletLoss(positive_mining_strategy='easy',
                      negative_mining_strategy='easy')
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss > 0


def test_triplet_loss_semi_hard():
    num_inputs = 10
    # y_true: labels
    y_true = tf.random.uniform((num_inputs,), 0, 3, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 16), 0, 1)
    tpl = TripletLoss(positive_mining_strategy='easy',
                      negative_mining_strategy='semi-hard',
                      reducer='sum')
    # y_true, y_preds
    loss = tpl(y_true, y_preds)
    assert loss
