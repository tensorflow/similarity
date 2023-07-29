import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.framework import combinations
from tensorflow.keras.losses import Reduction
from tensorflow_similarity import losses
from . import utils

@combinations.generate(combinations.combine(mode=["graph", "eager"]))
class TestLiftedStructLoss(tf.test.TestCase, parameterized.TestCase):
    def test_config(self):
        lifted_obj = losses.LiftedStructLoss(
            reduction=Reduction.SUM,
            name="lifted_loss",
        )
        self.assertEqual(lifted_obj.distance.name, "cosine")
        self.assertEqual(lifted_obj.name, "lifted_loss")
        self.assertEqual(lifted_obj.reduction, Reduction.SUM)

    @parameterized.named_parameters(
        {"testcase_name": "_fixed_margin", "margin": 1.1, "expected_loss": 157.68167},
    )
    def test_all_correct_unweighted(self, margin, expected_loss):
        """Tests the LiftedStructLoss with different parameters."""
        y_true, y_preds = utils.generate_perfect_test_batch()

        lifted_obj = losses.LiftedStructLoss(reduction=Reduction.SUM, margin=margin)
        loss = lifted_obj(y_true, y_preds)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    @parameterized.named_parameters(
        {"testcase_name": "_fixed_margin", "margin": 1.0, "expected_loss": 187.37393},
    )
    def test_all_mismatch_unweighted(self, margin, expected_loss):
        """Tests the LiftedStructLoss with different parameters."""
        y_true, y_preds = utils.generate_bad_test_batch()

        lifted_obj = losses.LiftedStructLoss(reduction=Reduction.SUM, margin=margin)
        loss = lifted_obj(y_true, y_preds)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    @parameterized.named_parameters(
        {"testcase_name": "_fixed_margin", "margin": 1.0, "expected_loss": 2.927718},
    )
    def test_no_reduction(self, margin, expected_loss):
        """Tests the LiftedStructLoss with different parameters."""
        y_true, y_preds = utils.generate_bad_test_batch()

        lifted_obj = losses.LiftedStructLoss(reduction=Reduction.NONE, margin=margin)
        loss = lifted_obj(y_true, y_preds)
        loss = self.evaluate(loss)
        expected_loss = self.evaluate(tf.fill(y_true.shape, expected_loss))
        self.assertArrayNear(loss, expected_loss, 0.001)

    @parameterized.named_parameters(
        {"testcase_name": "_fixed_margin", "margin": 1.0, "expected_loss": 2.414156913757324 },
    )
    def test_sum_reduction(self, margin, expected_loss):
        """Tests the LiftedStructLoss with different parameters."""
        y_true, y_preds = utils.generate_perfect_test_batch()

        lifted_obj = losses.LiftedStructLoss(reduction=Reduction.SUM, margin=margin)
        loss = lifted_obj(y_true, y_preds)
        expected_loss = y_true.shape[0] * expected_loss
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)