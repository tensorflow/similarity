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
from __future__ import annotations

from typing import Any

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Similarity")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A cosine decay LearningRateSchedule with a linear warmup period.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a linear warmup from 0.0
    to a max learning rate followed by a cosine decay function. It requires
    a `step` value to compute the decayed learning rate. You can just pass
    a TensorFlow variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a decayed learning rate
    when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer
    functions.

    It is computed as:

    ```python
    def decayed_learning_rate(step):
      step = min(step, total_steps)
      if step < warmup_steps:
        decayed = step / warmup_steps
      else:
        decay_steps = total_steps - warmup_steps
        cosine_decay = 0.5 * (1. + cos(pi * (step - warmup_steps) / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha

      return max_learning_rate * decayed
    ```

    Example usage:
    ```python
    total_steps = 1000
    warmup_steps = 100
    lr_decayed_fn = tf.keras.optimizers.schedules.WarmupCosineDecay(
        max_learning_rate, total_steps, warmup_steps)
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `max_learning_rate`.
    """

    def __init__(
        self,
        max_learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        alpha: float = 0.0,
        name: str | None = None,
    ):
        """Applies cosine decay with warmp up to the learning rate.

        Args:
          max_learning_rate: The max learning rate after warmup.
          total_steps: Total number of steps in the schedule.
          warmup_steps: Number of steps to warmup over. Must be smaller than the total number of steps.
          alpha: Minimum learning rate value as a fraction of initial_learning_rate.
          name: Optional name of the operation. Defaults to 'WarmupCosineDecay'.
        """
        super().__init__()

        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be less than the total steps")

        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha
        self.name = name

        self.max_learning_rate_tf = tf.convert_to_tensor(self.max_learning_rate, name="max_learning_rate")
        self.dtype = self.max_learning_rate_tf.dtype
        self.warmup_steps_tf = tf.cast(self.warmup_steps, self.dtype)

        self.cosine_decay = tf.keras.experimental.CosineDecay(max_learning_rate, total_steps - warmup_steps, alpha)

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            step = tf.cast(step, self.dtype)

            learning_rate = tf.cond(
                tf.math.less(step, self.warmup_steps_tf),
                lambda: tf.math.divide_no_nan(step, self.warmup_steps_tf) * self.max_learning_rate_tf,
                lambda: self.cosine_decay(step - self.warmup_steps_tf),
            )

            return learning_rate

    def get_config(self) -> dict[str, Any]:
        return {
            "max_learning_rate": self.max_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
            "name": self.name,
        }
