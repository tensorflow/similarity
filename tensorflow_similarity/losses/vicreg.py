# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VicReg Loss.
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning
    https://arxiv.org/abs/2105.04906
"""
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class VicReg(tf.keras.losses.Loss):
    """VicReg Loss.

    [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)
    """  # noqa

    def __init__(
        self,
        std_const: float = 1e-4,
        lambda_: float = 25,
        mu: float = 25,
        nu: float = 1,
        reduction: Callable = tf.keras.losses.Reduction.NONE,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.std_const = std_const
        self.reduction = reduction

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
        sim_loss = tf.keras.losses.MeanSquaredError(reduction=self.reduction)(za, zb)
        sim_loss = tf.keras.losses.MeanSquaredError(reduction="none")(za, zb)

        za = self.mean_center_columns(za)
        zb = self.mean_center_columns(zb)

        # std loss to maximize variance(information)
        std_za = tf.sqrt(tf.math.reduce_variance(za, 0) + self.std_const)
        std_zb = tf.sqrt(tf.math.reduce_variance(zb, 0) + self.std_const)

        std_loss_za = tf.reduce_mean(tf.math.maximum(0.0, 1 - std_za))
        std_loss_zb = tf.reduce_mean(tf.math.maximum(0.0, 1 - std_zb))

        std_loss = std_loss_za / 2 + std_loss_zb / 2

        off_diag_ca = self.cov_loss_each(za, batch_size)
        off_diag_cb = self.cov_loss_each(zb, batch_size)

        # covariance loss(1d tensor) for redundancy reduction
        cov_loss = off_diag_ca + off_diag_cb

        loss: FloatTensor = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * cov_loss

        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "std_const": self.std_const,
            "lambda_": self.lambda_,
            "mu": self.mu,
            "nu": self.nu,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def off_diagonal(self, x: FloatTensor) -> FloatTensor:
        n = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1])[:-1]
        off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
        off_diag: FloatTensor = tf.reshape(off_diagonals, [-1])
        return off_diag

    def cov_loss_each(self, z, batch_size):
        # cross-correlation matrix axa
        c = tf.matmul(z, z, transpose_a=True)
        c = c / tf.cast(batch_size - 1, dtype="float32")

        num_features = tf.shape(c)[0]

        off_diag_c = self.off_diagonal(c)
        off_diag_c = tf.math.pow(off_diag_c, 2)

        off_diag_c = tf.math.reduce_sum(off_diag_c) / tf.cast(num_features, tf.float32)

        return off_diag_c

    def mean_center_columns(self, x: FloatTensor) -> FloatTensor:
        col_mean = tf.math.reduce_mean(x, axis=0)

        norm_col: FloatTensor = x - col_mean
        return norm_col
