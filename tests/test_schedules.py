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
import math

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras.optimizers.schedules import learning_rate_schedule
from tensorflow.python.framework import combinations

from tensorflow_similarity.schedules import WarmupCosineDecay


def _maybe_serialized(lr_decay, serialize_and_deserialize):
    if serialize_and_deserialize:
        serialized = learning_rate_schedule.serialize(lr_decay)
        return learning_rate_schedule.deserialize(serialized)
    else:
        return lr_decay


def np_warmup_cosine_decay(*, step, max_lr, total_steps, warmup_steps, alpha=0.0):
    step = min(step, total_steps)
    if step < warmup_steps:
        decayed = step / warmup_steps
    else:
        decay_steps = total_steps - warmup_steps
        completed_fraction = (step - warmup_steps) / decay_steps
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
        decayed = (1.0 - alpha) * cosine_decay + alpha
    return max_lr * decayed


@combinations.generate(combinations.combine(mode=["graph", "eager"], serialize=[False, True]))
class WarmupCosineDecayTest(tf.test.TestCase, parameterized.TestCase):
    def testDecay(self, serialize):
        num_training_steps = 1000
        warmup_steps = 500
        max_lr = 1.0
        for step in range(0, 1500, 250):
            decayed_lr = WarmupCosineDecay(max_lr, num_training_steps, warmup_steps)
            decayed_lr = _maybe_serialized(decayed_lr, serialize)
            expected = np_warmup_cosine_decay(
                step=step,
                max_lr=max_lr,
                total_steps=num_training_steps,
                warmup_steps=warmup_steps,
            )
            self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

    def testAlpha(self, serialize):
        num_training_steps = 1000
        warmup_steps = 500
        max_lr = 1.0
        alpha = 0.1
        for step in range(0, 1500, 250):
            decayed_lr = WarmupCosineDecay(max_lr, num_training_steps, warmup_steps, alpha)
            decayed_lr = _maybe_serialized(decayed_lr, serialize)
            expected = np_warmup_cosine_decay(
                step=step,
                max_lr=max_lr,
                total_steps=num_training_steps,
                warmup_steps=warmup_steps,
                alpha=alpha,
            )
            self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

    def testFloat64InitLearningRate(self, serialize):
        num_training_steps = 1000
        warmup_steps = 500
        max_lr = np.float64(1.0)
        for step in range(0, 1500, 250):
            decayed_lr = WarmupCosineDecay(max_lr, num_training_steps, warmup_steps)
            decayed_lr = _maybe_serialized(decayed_lr, serialize)
            expected = np_warmup_cosine_decay(
                step=step,
                max_lr=max_lr,
                total_steps=num_training_steps,
                warmup_steps=warmup_steps,
            )
            self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

    def testMaxWarmupSteps(self, serialize):
        num_training_steps = 1000
        warmup_steps = 1000
        max_lr = 1.0
        msg = "warmup_steps must be less than the total steps"
        with self.assertRaisesRegex(ValueError, msg):
            _ = WarmupCosineDecay(max_lr, num_training_steps, warmup_steps)


if __name__ == "__main__":
    tf.test.main()
