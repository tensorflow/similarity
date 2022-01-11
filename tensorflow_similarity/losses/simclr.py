from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.keras.losses import Loss

from tensorflow_similarity.types import FloatTensor

LARGE_NUM = 1e9


@tf.keras.utils.register_keras_serializable(package="Similarity")
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
        margin: float = 0.001,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.temperature = tf.constant(temperature, dtype="float32")
        self.use_hidden_norm = use_hidden_norm
        self.margin = margin

    @tf.function
    def call(self, za: FloatTensor, zb: FloatTensor) -> FloatTensor:
        """Compute the lost.

        Args:
            za: Embedding A
            zb: Embedding B

        Returns:
            loss
        """
        # compute the diagonal
        batch_size = tf.shape(za)[0]
        diag = tf.one_hot(tf.range(batch_size), batch_size)

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
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

        # 1D tensor
        per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels, distances
        )

        # 1D tensor
        loss: FloatTensor = per_example_loss * 0.5 + self.margin

        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "temperature": self.temperature,
            "use_hidden_norm": self.use_hidden_norm,
            "margin": self.margin,
        }
        base_config = super().get_config()
        return {**base_config, **config}
