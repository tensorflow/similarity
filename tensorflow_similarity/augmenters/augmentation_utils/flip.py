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
from __future__ import annotations

import tensorflow as tf

from tensorflow_similarity.augmenters.augmentation_utils.random_apply import (
    random_apply,
)
from tensorflow_similarity.types import Tensor


def random_random_flip_left_right(image: Tensor, p: float = 0.5) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        return tf.image.random_flip_left_right(image)

    return random_apply(_transform, p=p, x=image)


def random_random_flip_top_bottom(image: Tensor, p: float = 0.5) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        return tf.image.random_flip_up_down(image)

    return random_apply(_transform, p=p, x=image)
