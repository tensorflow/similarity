import tensorflow as tf
from typing import Any, Dict
from tensorflow_similarity.types import FloatTensor
from tensorflow.keras.losses import Loss

LARGE_NUM = 1e9

# FIXME: make sure to register
# @tf.keras.utils.register_keras_serializable(package="Similarity")


class SimCLRLoss(Loss):
    """SimCLR Loss
    # FIXME original reference
    used in [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)
    code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)
    """
    def __init__(self,
                 temperature: float = 1.0,
                 use_hidden_norm: bool = True,
                 **kwargs):
        super().__init__(kwargs)
        self.temperature = tf.constant(temperature, dtype='float32')
        self.use_hidden_norm = use_hidden_norm

    def call(self,
             hs: FloatTensor,
             zs: FloatTensor) -> FloatTensor:
        """compute the loast.
        Args:
        h: Encoders outputs
        z: Projectors outputs

        Returns:
        loss
        """
        # unpack
        za, zb = zs

        # compute the diagonal
        batch_size = tf.size(za)
        diag = tf.linalg.diag(tf.ones(batch_size, dtype=tf.float32))

        if self.use_hidden_norm:
            za = tf.math.l2_normalize(za)
            zb = tf.math.l2_normalize(zb)

        # compute pairwise
        ab = tf.matmul(za, zb, transpose_b=True)
        aa = tf.matmul(za, za, transpose_b=True)

        # divide by temperature once to go faster
        ab = ab / self.temperature
        aa = aa / self.temperature

        # set the diagonal to a large negative number to ensure that z_i * z_i
        # is close to zero in the cross entropy.
        aa = aa - diag * LARGE_NUM

        distances = tf.concat((ab, aa), axis=1)

        per_example_loss: FloatTensor = tf.nn.softmax_cross_entropy_with_logits(diag, distances)

        return per_example_loss

    def to_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "use_hidden_norm": self.use_hidden_norm
        }
