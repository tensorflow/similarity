from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from tensorflow.keras.losses import Loss

from tensorflow_similarity.types import FloatTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class VicReg(Loss):
    """VicReg Loss"""

    def __init__(self,
                 const_std: float = 1e-4
                 lambda_: float = 25,
                 mu: float = 25,
                 nu: float = 1,
                 reduction: Callable = tf.keras.losses.Reduction.AUTO,
                 name: Optional[str] = None,
                 **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu

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
         
        # distance loss to measure similarity between representations
        sim_loss = tf.keras.losses.MeanSquaredError()(z_a, z_b)
        
        # std loss to maximize variance(information)
        std_za = tf.sqrt(tf.math.reduce_variance(za, 0) + self.std_const)
        std_zb = tf.sqrt(tf.math.reduce_variance(zb, 0) + self.std_const)
        std_loss_za = tf.reduce_mean(tf.max(0, 1 - std_za))
        std_loss_zb = tf.reduce_mean(tf.max(0, 1 - std_zb))
        std_loss = std_loss_za + std_loss_zb
        
        za = self.standardize_columns(za)
        zb = self.standardize_columns(zb)

        # cross-correlation matrix axa
        ca = tf.matmul(za, za, transpose_a=True)
        ca = ca / tf.cast(batch_size-1, dtype="float32")
        
        # cross-correlation matrix bxb
        cb = tf.matmul(zb, zb, transpose_a=True)
        cb = cb / tf.cast(batch_size-1, dtype="float32")

    
        off_diag_ca = self.off_diagonal(c)
        off_diag_ca = tf.math.pow(off_diag_ca, 2)
        off_diag_ca = tf.math.reduce_sum(off_diag_ca)
        
        
        off_diag_cb = self.off_diagonal(c)
        off_diag_cb = tf.math.pow(off_diag_cb, 2)
        off_diag_cb = tf.math.reduce_sum(off_diag_cb)

        # covariance loss(1d tensor) for redundancy reduction
        cov_loss: FloatTensor = off_diag_ca + off_diag_cb
        
        loss = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * cov_loss
        
        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "std_const": self.std_const
            "lambda_": self.lambda_,
            "mu": self.mu,
            "nu": self.nu
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def off_diagonal(self, x: FloatTensor) -> FloatTensor:
        n = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1])[:-1]
        off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
        off_diag: FloatTensor = tf.reshape(off_diagonals, [-1])
        return off_diag

    def standardize_columns(self, x: FloatTensor) -> FloatTensor:
        col_mean = tf.math.reduce_mean(x, axis=0)

        norm_col: FloatTensor = x - col_mean
        return norm_col
