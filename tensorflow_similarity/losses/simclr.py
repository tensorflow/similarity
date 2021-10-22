import tensorflow as tf
from typing import Any, Callable, Dict, Optional
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

    def __init__(
        self,
        temperature: float = 0.05,
        use_hidden_norm: bool = True,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.temperature = tf.constant(temperature, dtype="float32")
        self.use_hidden_norm = use_hidden_norm

    def call(self, va: FloatTensor, vb: FloatTensor) -> FloatTensor:
        """compute the loast.
        Args:
            va: View A

            vb: View B
        Returns:
            loss
        """
        # compute the diagonal
        batch_size = tf.shape(va)[0]
        diag = tf.one_hot(tf.range(batch_size), batch_size)

        if self.use_hidden_norm:
            va = tf.math.l2_normalize(va)
            vb = tf.math.l2_normalize(vb)

        # compute pairwise
        ab = tf.matmul(va, vb, transpose_b=True)
        aa = tf.matmul(va, va, transpose_b=True)

        # divide by temperature once to go faster
        ab = ab / self.temperature
        aa = aa / self.temperature

        # set the diagonal to a large negative number to ensure that z_i * z_i
        # is close to zero in the cross entropy.
        aa = aa - diag * LARGE_NUM

        distances = tf.concat((ab, aa), axis=1)
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

        per_example_loss: FloatTensor = tf.nn.softmax_cross_entropy_with_logits(
            labels, distances
        )

        return per_example_loss

    def to_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "use_hidden_norm": self.use_hidden_norm,
        }
