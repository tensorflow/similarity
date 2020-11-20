import tensorflow as tf


def true_positives(y_true, y_pred):
    """Calculates the number of true positives.

    Args:
        y_true (list(int)): array if class as int.
        y_pred (list(int)): array of class as int

    Returns:
        int: num true positives
    """
    tp = tf.cast(tf.math.equal(y_true, y_pred), 'int32')
    return tf.cast(tf.reduce_sum(tp), 'float32')


def false_positives(y_true, y_pred):
    """Calculates the number of false positives.

    Args:
        y_true (list(int)): array if class as int.
        y_pred (list(int)): array of class as int

    Returns:
        int: num true positives
    """
    tp = tf.cast(tf.math.not_equal(y_true, y_pred), 'int32')
    return tf.cast(tf.reduce_sum(tp), 'float32')


def accuracy(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    return tp / len(y_true)


@tf.function
def precision(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return (tp / (tp + fp))
