import tensorflow as tf


def generate_perfect_test_batch(batch_size=32):
    """Generate a batch of embeddings and labels that will result in a perfect loss score."""

    # y_true: labels
    y_true = tf.range(0, batch_size, dtype=tf.int32)
    y_true = tf.concat([y_true, y_true], axis=0)

    # y_preds: embedding
    y_preds = tf.one_hot(y_true, depth=batch_size, dtype=tf.float32)

    y_true = tf.expand_dims(y_true, axis=1)
    return y_true, y_preds


def generate_bad_test_batch(batch_size=32):
    """Generate a batch of embeddings and labels that will result in a mismatch for all classes."""

    # y_true: labels
    y_true = tf.range(0, batch_size, dtype=tf.int32)

    # y_preds: embedding
    y_preds = tf.concat([y_true, y_true[::-1]], axis=0)
    y_preds = tf.one_hot(y_preds, depth=batch_size, dtype=tf.float32)

    y_true = tf.concat([y_true, y_true], axis=0)
    y_true = tf.expand_dims(y_true, axis=1)
    return y_true, y_preds
