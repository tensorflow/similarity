import numpy as np
import tensorflow as tf

from tensorflow_similarity.matchers import MatchMajorityVote


def test_predict():
    mn = MatchMajorityVote()

    lookup_labels = tf.constant([
        [10, 12, 10, 12, 10],
        # Ties should take the closer label.
        [20, 13, 13, 20, 30]]
    )
    lookup_distances = tf.constant([
        [1., 1.1, 1.2, 1.3, 1.4],
        [2., 2.1, 2.2, 2.3, 2.4]]
    )

    pred_labels, pred_dist = mn.predict(lookup_labels, lookup_distances)

    np.testing.assert_array_equal(
            pred_labels.numpy(),
            np.array([[10], [20]]))
    np.testing.assert_allclose(
            pred_dist.numpy(),
            np.array([[1.2], [2.2]]))
