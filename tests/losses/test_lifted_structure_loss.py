import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras.losses import Reduction
from tensorflow.python.framework import combinations
from tensorflow_similarity import losses
from . import utils

@combinations.generate(combinations.combine(mode=["graph", "eager"]))
class LiftedStructureLossTest(tf.test.TestCase, parameterized.TestCase):
    def test_config(self):
        lsl_obj = losses.LiftedStructLoss( name="lifted_struct_loss", distance="cosine"
        )
        self.assertEqual(lsl_obj.distance.name, "cosine")
        self.assertEqual(lsl_obj.name, "lifted_struct_loss")
    
    # TODO calculate results by hand before
