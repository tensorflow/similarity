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


def precision(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return (tp / (tp + fp))


def recall(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    return (tp / len(y_true))


def f1_score(y_true, y_preds):
    """Compute the F1 score, also known as balanced F-score or F-measure

  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    Args:
        y_true (1D tensor(int)): ground truth.
        y_pred (1D tensor(int)): Estimated targets as returned by a classifier.

    Returns:
        float: f1 score
    """
    pr = precision(y_true, y_preds)
    re = recall(y_true, y_preds)
    return 2 * (pr * re) / (pr + re)


def fast_f1_score(precision, recall):
    """Compute the F1 score, also known as balanced F-score or F-measure
    when precision and recall are already computed

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    Args:
        precision (float): Precision metric
        recall (float): Precision metric

    Returns:
        float: f1 score
    """
    return 2 * (precision * recall) / (precision + recall)
