import numpy as np
import tensorflow as tf

from tensorflow_similarity.matchers import MatchNearest


def test_predict():
    mn = MatchNearest()

    lookup_labels = (
            tf.constant([[10, 12], [20, 13], [30, 14], [40, 15]]))
    lookup_distances = (
            tf.constant([[1., 1.5], [1., 1.7], [2., 2.1], [2., 2.2]]))

    pred_labels, pred_dist = mn.predict(lookup_labels, lookup_distances)

    np.testing.assert_array_equal(
            pred_labels.numpy(),
            np.array([[10], [20], [30], [40]]))
    np.testing.assert_array_equal(
            pred_dist.numpy(),
            np.array([[1.], [1.], [2.], [2.]]))
