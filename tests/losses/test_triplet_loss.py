import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.losses import Reduction
from tensorflow.python.framework import combinations

from tensorflow_similarity import losses

from . import utils


@combinations.generate(combinations.combine(mode=["graph", "eager"]))
class TripletLossTest(tf.test.TestCase, parameterized.TestCase):
    def test_config(self):
        tpl_obj = losses.TripletLoss(reduction=Reduction.SUM, name="triplet_loss", distance="cosine")
        self.assertEqual(tpl_obj.distance.name, "cosine")
        self.assertEqual(tpl_obj.name, "triplet_loss")
        self.assertEqual(tpl_obj.reduction, Reduction.SUM)

    @parameterized.named_parameters(
        {"testcase_name": "_soft_margin", "margin": None, "expected_loss": 0.31326169},
        {"testcase_name": "_fixed_margin", "margin": 1.1, "expected_loss": 0.1},
    )
    def test_all_correct_unweighted(self, margin, expected_loss):
        """Test that assumes a perfect embedding.

        Using cosine distance, the prefect embedding will contain one hot encoded
        embeddings with a value of 1.0 in the class position and 0 everywhere else.

        Using Soft Margin we expect `pos_dist - neg_dist = -1.0` yielding
        `ln(1 + e^(-1.0)) = 0.31326169`.

        Using a fixed margin of 1.1 we expect the loss to stop at 0.1.
        """
        tpl_obj = losses.TripletLoss(reduction=Reduction.SUM_OVER_BATCH_SIZE, margin=margin)
        y_true, y_preds = utils.generate_perfect_test_batch()
        loss = tpl_obj(y_true, y_preds)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    @parameterized.named_parameters(
        {"testcase_name": "_soft_margin", "margin": None, "expected_loss": 0.69314718},
        {"testcase_name": "_fixed_margin", "margin": 1.0, "expected_loss": 1.0},
    )
    def test_all_mismatch_unweighted(self, margin, expected_loss):
        """Test that assumes none of the classes match.

        Using cosine distance, each embedding is one hot encoded with a value of
        1.0 in the class position and 0 everywhere else.

        Using Soft Margin we expect `pos_dist - neg_dist = 0.0` yielding
        `ln(1 + e^(0.0)) = 0.69314718`.

        Using a fixed margin of 1.0 we expect the loss to stop at 1.0.
        """
        tpl_obj = losses.TripletLoss(reduction=Reduction.SUM_OVER_BATCH_SIZE, margin=margin)
        y_true, y_preds = utils.generate_bad_test_batch()
        loss = tpl_obj(y_true, y_preds)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    @parameterized.named_parameters(
        {"testcase_name": "_soft_margin", "margin": None, "expected_loss": 0.31326169},
        {"testcase_name": "_fixed_margin", "margin": 1.1, "expected_loss": 0.1},
    )
    def test_no_reduction(self, margin, expected_loss):
        tpl_obj = losses.TripletLoss(reduction=Reduction.NONE, margin=margin)
        y_true, y_preds = utils.generate_perfect_test_batch()
        loss = tpl_obj(y_true, y_preds)
        loss = self.evaluate(loss)
        expected_loss = self.evaluate(tf.fill(y_true.shape, expected_loss))
        self.assertArrayNear(loss, expected_loss, 0.001)

    @parameterized.named_parameters(
        {"testcase_name": "_soft_margin", "margin": None, "expected_loss": 0.31326169},
        {"testcase_name": "_fixed_margin", "margin": 1.1, "expected_loss": 0.1},
    )
    def test_sum_reduction(self, margin, expected_loss):
        tpl_obj = losses.TripletLoss(reduction=Reduction.SUM, margin=margin)
        y_true, y_preds = utils.generate_perfect_test_batch()
        loss = tpl_obj(y_true, y_preds)
        expected_loss = y_true.shape[0] * expected_loss
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    @parameterized.named_parameters(
        {
            "testcase_name": "_easy",
            "positive_mining_strategy": "easy",
            "negative_mining_strategy": "easy",
            "expected_loss": 0.333,
        },
        {
            "testcase_name": "_hard",
            "positive_mining_strategy": "hard",
            "negative_mining_strategy": "hard",
            "expected_loss": 2.0,
        },
    )
    def test_mining(self, positive_mining_strategy, negative_mining_strategy, expected_loss):
        """Test the easy mining strategy

        Easy mining strategy will always pick the minimum distance for the
        positive or negative pair. This means that the pair for positive
        should always be [1.0, 0.0] and for negative should always be [0.0, 1.0].
        This means that the loss should be
        sum([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]) / 6 = 0.333.

        Hard ming strategy will always pick the maximum distance for the
        positive or negative pair. This means that the pair for positive
        should always be [0.0, 1.0] and for negative should always be [1.0, 0.0].
        This means that the loss should be
        sum([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 6 = 2.0.
        """
        tpl_obj = losses.TripletLoss(
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
            positive_mining_strategy=positive_mining_strategy,
            negative_mining_strategy=negative_mining_strategy,
            margin=1.0,
        )
        labels = tf.expand_dims(tf.constant([0, 0, 0, 1, 1, 1], dtype=(tf.int32)), axis=1)
        embeddings = tf.constant(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=(tf.float32)
        )
        loss = tpl_obj(labels, embeddings)
        self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

    def test_semi_hard_mining(self):
        """Test the easy positive mining and semi-hard negative mining strategy.

        Easy positive mining strategy will just pick the other positive example,
        i.e., the pos_idxs will be [1,0,3,2]

        Semi-hard negative mining strategy will pick the closest negative example
        that is > the max distance between the positive pairs, if one exists,
        otherwise it will pick the maximal negative example. The neg_idxs will be
        [3,2,1,1] where the second positive example actully has no negatives that
        are > the max distance between the maximal positive.This is why the loss
        for the second positive example is > 1.0.

        The expected pairwise distances are:
            [[0.0,  0.5,  1.0,     0.98],   # Choose 0.98 because it's > 0.5 but < 1.0
             [0.5,  0.0,  0.13,    0.12],   # Choose 0.13 because it's there is no > 0.5
             [1.0,  0.13, 0.0,     0.00005],# Choose 0.13 because it's > 5e-5 but < 1.0
             [0.98, 0.12, 0.00005, 0.0]]    # Choose 0.12 because it's > 5e-5 but < 0.98
        """
        tpl_obj = losses.TripletLoss(
            reduction=Reduction.NONE,
            positive_mining_strategy="easy",
            negative_mining_strategy="semi-hard",
            margin=1.0,
        )
        labels = tf.expand_dims(tf.constant([0, 0, 1, 1], dtype=(tf.int32)), axis=1)
        embeddings = tf.constant(
            [[1.0, 0.0], [0.5, 0.8660254], [0.0, 1.0], [0.01010049, 0.999949]], dtype=(tf.float32)
        )
        loss = tpl_obj(labels, embeddings)
        expected_loss = tf.constant([[0.51], [1.366], [0.866], [0.871]], dtype=(tf.float32))
        self.assertArrayNear(self.evaluate(loss), self.evaluate(expected_loss), 0.001)
