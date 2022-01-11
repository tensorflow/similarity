from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.keras.losses import Loss

from tensorflow_similarity.types import FloatTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class Barlow(Loss):
    """Barlow Loss"""

    def __init__(self,
                 lambda_: float = 5e-3,
                 margin: float = 1e-12,
                 reduction: Callable = tf.keras.losses.Reduction.AUTO,
                 name: Optional[str] = None,
                 **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.lambda_ = lambda_
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

        za_norm = self.normalize_columns(za)
        zb_norm = self.normalize_columns(zb)

        # compute pairwise
        c = tf.matmul(za_norm, zb_norm, transpose_a=True)
        c = c / tf.cast(batch_size, dtype="float32")

        on_diag = 1.0 - tf.linalg.diag_part(c)
        on_diag = tf.math.pow(on_diag, 2)
        on_diag = tf.math.reduce_sum(on_diag)

        off_diag = self.off_diagonal(c)
        off_diag = tf.math.pow(off_diag, 2)
        off_diag = tf.math.reduce_sum(off_diag)

        # 1D Tensor
        loss: FloatTensor = off_diag * self.lambda_ + on_diag + self.margin

        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "lambda_": self.lambda_,
            "margin": self.margin,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def off_diagonal(self, x: FloatTensor) -> FloatTensor:
        n = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1])[:-1]
        off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
        off_diag: FloatTensor = tf.reshape(off_diagonals, [-1])
        return off_diag

    def normalize_columns(self, x: FloatTensor) -> FloatTensor:
        col_mean = tf.math.reduce_mean(x, axis=0)
        col_std = tf.math.reduce_std(x, axis=0)

        norm_col: FloatTensor = tf.math.divide_no_nan((x - col_mean), col_std)
        return norm_col
