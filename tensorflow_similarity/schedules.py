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
from typing import Any, Dict

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with a warmup period.

    This learning rate schedule is useful for training when using the Barlow Twin Loss.

    The warmup period applies a linear scaling to the CosineDecay schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        warmup_steps: int,
        warmup_learning_rate: float = 0.0,
        alpha: float = 0.0,
        name: str = "WarmUpCosine",
    ):
        """Applies cosine decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to warmup over. Must be smaller than the number of
            decay_steps.
          warmup_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial warmup learning rate. Must be smaller than
            the initial_learning_rate. Defaults to 0.0.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of initial_learning_rate.
            Defaults to 0.0.
          name: String. Optional name of the operation. Defaults to 'WarmUpCosine'.
        """

        super().__init__()

        if warmup_learning_rate > initial_learning_rate:
            raise ValueError("warmup_learning_rate must be smaller than the initial_learning_rate")

        if warmup_steps > decay_steps:
            raise ValueError("warmup_steps must be smaller than the decay_steps")
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.name = name

        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha,
        )
        # Compute the warmup increment.
        self.tf_initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
        self.dtype = self.tf_initial_learning_rate.dtype
        self.learning_rate_delta = tf.convert_to_tensor(
            self.warmup_learning_rate / self.initial_learning_rate, self.dtype
        )
        self.warmup_inc = tf.math.divide_no_nan(
            (1.0 - self.learning_rate_delta),
            tf.convert_to_tensor(self.warmup_steps, self.dtype),
        )

        # If the warmup increment is zero we have no warm up phase and we set
        # the learning rate delta to 1.0 to ensure the warmup_scaler value is
        # always fixed at 1.0.
        if self.warmup_inc == 0:
            self.learning_rate_delta = tf.constant([1.0], self.dtype)

    def __call__(self, step: FloatTensor) -> FloatTensor:
        global_step_recomp = tf.cast(step, self.dtype)
        warmup_scaler = tf.minimum(1.0, self.warmup_inc * global_step_recomp + self.learning_rate_delta)
        learning_rate: FloatTensor = self.cosine_decay(global_step_recomp) * warmup_scaler
        return learning_rate

    def get_config(self) -> Dict[str, Any]:
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }
