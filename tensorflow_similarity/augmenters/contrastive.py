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

import os
from collections.abc import Callable

import tensorflow as tf

from tensorflow_similarity.augmenters import Augmenter
from tensorflow_similarity.types import Tensor


class ContrastiveAugmenter(Augmenter):
    def __init__(self, process: Callable, num_cpu: int | None = os.cpu_count()):
        self.process = process
        self.num_cpu = num_cpu

    def augment(self, x: Tensor, y: Tensor, num_views: int, is_warmup: bool) -> list[Tensor]:
        with tf.device("/cpu:0"):
            inputs = tf.stack(x)

            views = []
            for _ in range(num_views):
                # multi-cor augementations
                view = tf.map_fn(self.process, inputs, parallel_iterations=self.num_cpu)
                views.append(view)
            return views
