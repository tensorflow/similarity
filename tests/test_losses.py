import tensorflow as tf
from tensorflow_similarity.losses import TripletLoss


def test_triplet_loss():
    num_inputs = 100
    # y_true: labels
    y_true = tf.random.uniform((num_inputs,), 0, 10, dtype=tf.int32)
    # y_preds: embedding
    y_preds = tf.random.uniform((num_inputs, 20), 0, 1)
    tpl = TripletLoss()
    # y_true, y_preds
    loss = tpl.__call__(y_true, y_preds)
    assert loss > 0.0
