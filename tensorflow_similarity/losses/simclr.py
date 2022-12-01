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
from __future__ import annotations

from typing import Any

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class SimCLRLoss(tf.keras.losses.Loss):
    """SimCLR Loss.
    [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
    code adapted from [original github](https://github.com/google-research/simclr/tree/master/tf2)
    """

    LARGE_NUM = 1e9

    def __init__(self, temperature: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def contrast(self, hidden1: FloatTensor, hidden2: FloatTensor) -> FloatTensor:

        # local replica batch size
        batch_size = tf.shape(hidden1)[0]

        if not tf.distribute.in_cross_replica_context():
            # SimCLR loss computes similarity with 2N-1 other examples, when N is the global_batch_size.
            # In distributed training, each replica sees only N/n_replicas examples and to compare with all 2N-1
            # examples we must first get them from other replicas.
            strategy = tf.distribute.get_strategy()
            hidden1_large = SimCLRLoss.cross_replica_concat(hidden1, strategy)
            hidden2_large = SimCLRLoss.cross_replica_concat(hidden2, strategy)
            enlarged_batch_size = tf.shape(hidden1_large)[0]
            replica_context = tf.distribute.get_replica_context()
            labels_idx = tf.range(batch_size) + replica_context.replica_id_in_sync_group * batch_size
            labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
            masks = tf.one_hot(labels_idx, enlarged_batch_size)
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
            masks = tf.one_hot(tf.range(batch_size), batch_size)

        # compute pairwise
        ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / self.temperature
        aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / self.temperature

        # set the diagonal to a large negative number to ensure that z_i * z_i
        # is close to zero in the cross entropy.
        aa = aa - masks * SimCLRLoss.LARGE_NUM

        distances = tf.concat((ab, aa), axis=1)

        # 1D tensor
        per_example_loss: FloatTensor = tf.nn.softmax_cross_entropy_with_logits(labels, distances)

        return per_example_loss

    def call(self, za: FloatTensor, zb: FloatTensor) -> FloatTensor:
        """Compute the loss.

        Args:
            za: Embedding A
            zb: Embedding B

        Returns:
            loss: Per replica loss
        """
        # We expect za and zb to be rank 2 tensors.
        za = tf.math.l2_normalize(za, axis=1)
        zb = tf.math.l2_normalize(zb, axis=1)

        loss_a = self.contrast(za, zb)
        loss_b = self.contrast(zb, za)

        # When used inside of built-in training loops such as Model.fit, tf.keras.losses.Reduction
        # will default to SUM_OVER_BATCH_SIZE which takes into account the global batch size across all GPUs.
        loss: FloatTensor = loss_a + loss_b
        return loss

    @staticmethod
    def cross_replica_concat(tensor, strategy):
        """Reduce a concatenation of the `tensor` across TPU cores.
        Args:
            tensor: tensor to concatenate.
            strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.
        Returns:
            Tensor of the same rank as `tensor` with first dimension `num_replicas`
            times larger.
        """
        num_replicas = strategy.num_replicas_in_sync

        replica_context = tf.distribute.get_replica_context()
        with tf.name_scope("cross_replica_concat"):
            # This creates a tensor that is like the input tensor but has an added
            # replica dimension as the outermost dimension. On each replica it will
            # contain the local values and zeros for all other values that need to be
            # fetched from other replicas.
            ext_tensor = tf.scatter_nd(
                indices=[[replica_context.replica_id_in_sync_group]],
                updates=[tensor],
                shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0),
            )

            # As every value is only present on one replica and 0 in all others, adding
            # them all together will result in the full tensor on all replicas.
            ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, ext_tensor)

            # Flatten the replica dimension.
            # The first dimension size will be: tensor.shape[0] * num_replicas
            # Using [-1] trick to support also scalar input.
            return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

    def get_config(self) -> dict[str, Any]:
        config = {
            "temperature": self.temperature,
        }
        base_config = super().get_config()
        return {**base_config, **config}
