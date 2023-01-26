import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.losses import Reduction
from tensorflow.python.framework import combinations

from tensorflow_similarity import losses


@tf.function
def softnn_util(y_true, x, temperature=1):
    """
    A simple loop based implementation of soft
    nearest neighbor loss to test the code.
    https://arxiv.org/pdf/1902.01889.pdf
    """

    batch_size = tf.shape(y_true)[0]
    loss = 0.0
    for i in tf.range(batch_size, dtype=tf.int32):
        numerator = 0.0
        denominator = 0.0
        for j in tf.range(batch_size, dtype=tf.int32):
            if i == j:
                continue
            if y_true[i] == y_true[j]:
                numerator += tf.math.exp(-1 * tf.math.reduce_sum(tf.math.square(x[i] - x[j])) / temperature)
            denominator += tf.math.exp(-1 * tf.math.reduce_sum(tf.math.square(x[i] - x[j])) / temperature)
        if numerator == 0.0:
            continue
        loss += tf.math.log(numerator / denominator)
    return -loss / tf.cast(batch_size, tf.float32)


@combinations.generate(
    combinations.combine(
        mode=["graph", "eager"],
    )
)
class SoftNNLossTest(tf.test.TestCase, parameterized.TestCase):
    def test_config(self):
        softnn_obj = losses.SoftNearestNeighborLoss(
            reduction=Reduction.SUM,
            name="softnn_loss",
            distance="cosine",
        )
        self.assertEqual(softnn_obj.distance.name, "cosine")
        self.assertEqual(softnn_obj.name, "softnn_loss")
        self.assertEqual(softnn_obj.reduction, Reduction.SUM)

    @parameterized.parameters((0.1), (0.5), (1), (2), (5), (10), (50))
    def test_all_correct(self, temperature):
        num_inputs = 10
        n_classes = 10

        softnn_obj = losses.SoftNearestNeighborLoss(
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
            temperature=temperature,
        )

        # y_true: labels
        y_true = tf.random.uniform((num_inputs, 1), 0, n_classes, dtype=tf.int32)
        # x: embeddings
        y_preds = tf.random.uniform((num_inputs, 20), 0, 1)

        loss = softnn_obj(y_true, y_preds)
        loss_check = softnn_util(
            y_true,
            y_preds,
            temperature,
        )

        loss_diff = loss - loss_check
        self.assertLess(self.evaluate(loss_diff), 1e-3)
