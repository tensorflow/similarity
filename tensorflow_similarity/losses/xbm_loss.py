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
"""VicReg Loss
    Cross-batch memory for embedding learning.
    https://arxiv.org/abs/1912.06798
"""
from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from tensorflow_similarity.losses import MetricLoss
from tensorflow_similarity.types import FloatTensor


def _add_memory_variable(tensor):
    """Creates an empty variable with same shape and dtype as `tensor`."""
    shape_no_batch = tensor.shape[1:]
    dtype = tensor.dtype
    init_value = tf.constant([], shape=[0, *shape_no_batch], dtype=dtype)
    var = tf.Variable(
        initial_value=init_value,
        shape=[None, *shape_no_batch],
        dtype=dtype,
        trainable=False,
    )
    return var


@tf.keras.utils.register_keras_serializable(package="Similarity")
class XBM(MetricLoss):
    """Cross-batch memory wrapper for MetricLoss instances.

    Maintains a memory queue of past embedding batches. Batch embeddings are
    paired with all embeddings in the memory queue, and loss is calculated from
    these pairs.

    Reference

    Wang, Xun, et al. "Cross-batch memory for embedding learning."
    https://arxiv.org/abs/1912.06798

    Standalone usage:

    >>> loss = tensorflow_similarity.losses.MultiSimilarityLoss()
    >>> loss = XBM(loss, memory_size=1000, warmup_steps=100)
    >>> y_pred = tf.random.uniform(shape=[4, 16])
    >>> y_true = tf.constant([[1], [2], [2], [3]])
    >>> loss(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.3740389>

    Args:
      loss: MetricLoss instance to use for computing loss.
      memory_size: Integer specifying the number of past embeddings to
        maintain in the memory queue.
      warmup_steps: Integer specifying the number of warmup steps where
        loss is calculated without using the memory queue.

    Returns:
      Loss value for cross-batched pairs

    NOTE:
    This will trigger multiple tf.function retracings if called multiple
    times in Eager mode, because of the dynamic size of the memory queue.

    """

    def __init__(
        self,
        loss: MetricLoss,
        memory_size: int,
        warmup_steps: int = 0,
        reduction: Callable = tf.keras.losses.Reduction.AUTO,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(loss.fn, reduction, name, **kwargs)

        self.loss = loss
        self.distance = loss.distance
        self._fn_kwargs.update(loss._fn_kwargs)

        # Diagonal is removed from positive pair mask because the batch is
        # always contained in the beginning of the memory.
        self._fn_kwargs["remove_diagonal"] = True

        self.memory_size = memory_size
        self.warmup_steps = warmup_steps
        self._y_true_memory = None
        self._y_pred_memory = None
        self._total_steps = tf.Variable(0, dtype=tf.int64, trainable=False)

    def call(self, y_true: FloatTensor, y_pred: FloatTensor) -> FloatTensor:
        with tf.device("/cpu:0"):
            # Build memory from first batch
            if self._y_true_memory is None:
                self._y_true_memory = _add_memory_variable(y_true)
            if self._y_pred_memory is None:
                self._y_pred_memory = _add_memory_variable(y_pred)

        # Enqueue (concat batch to beginning of memory)
        y_true_mem = tf.concat([y_true, self._y_true_memory], axis=0)
        y_pred_mem = tf.concat([y_pred, self._y_pred_memory], axis=0)

        # Dequeue (truncate to memory size limit)
        y_true_mem = y_true_mem[: self.memory_size]
        y_pred_mem = y_pred_mem[: self.memory_size]

        def _xbm_step():
            # Update memory with new values
            self._y_true_memory.assign(y_true_mem)
            self._y_pred_memory.assign(y_pred_mem)
            return y_true_mem, y_pred_mem

        def _warmup_step():
            # Memory is not updated during warmup steps, so it remains empty.
            # Therefore this corresponds to the original batch
            return y_true_mem, y_pred_mem

        self._total_steps.assign_add(1)
        y_true_mem, y_pred_mem = tf.cond(
            self._total_steps > self.warmup_steps,
            true_fn=_xbm_step,
            false_fn=_warmup_step,
        )

        loss: FloatTensor = self.fn(y_true, y_pred, y_true_mem, y_pred_mem, **self._fn_kwargs)
        return loss

    def get_config(self) -> Dict[str, Any]:
        config = {
            "loss": tf.keras.utils.serialize_keras_object(self.loss),
            "memory_size": self.memory_size,
            "warmup_steps": self.warmup_steps,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MetricLoss:
        config["loss"] = tf.keras.utils.deserialize_keras_object(config["loss"])
        return cls(**config)
