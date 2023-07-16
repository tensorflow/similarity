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
            distance="cosine",
        )
        self.assertEqual(lifted_obj.distance.name, "cosine")
        self.assertEqual(lifted_obj.name, "lifted_loss")
        self.assertEqual(lifted_obj.reduction, Reduction.SUM)

    
    
