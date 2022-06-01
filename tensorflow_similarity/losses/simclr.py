# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SimCLR Loss.
    A Simple Framework for Contrastive Learning of Visual Representations
    https://arxiv.org/abs/2002.05709
"""
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor

LARGE_NUM = 1e9


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SimCLRLoss(tf.keras.losses.Loss):
    """SimCLR Loss.
    [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
    code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.001,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.temperature = temperature
        self.margin = margin

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

        # We expect za and zb to be rank 2 tensors.
        za = tf.math.l2_normalize(za, axis=1)
        zb = tf.math.l2_normalize(zb, axis=1)

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
            "margin": self.margin,
        }
        base_config = super().get_config()
        return {**base_config, **config}
