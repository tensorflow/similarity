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

from collections.abc import Callable, Sequence
from typing import TypeVar

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, IntTensor

from .memory_samplers import MultiShotMemorySampler
from .samplers import Augmenter

T = TypeVar("T", FloatTensor, IntTensor)


def load_image(path: str, target_size: tuple[int, int] | None = None) -> T:
    image_string = tf.io.read_file(path)
    image: T = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if target_size:
        image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.LANCZOS3)
        image = tf.clip_by_value(image, 0.0, 1.0)
    return image


class MultiShotFileSampler(MultiShotMemorySampler):
    def __init__(
        self,
        x,
        y,
        load_example_fn: Callable = load_image,
        classes_per_batch: int = 2,
        examples_per_class_per_batch: int = 2,
        steps_per_epoch: int = 1000,
        class_list: Sequence[int] | None = None,
        total_examples_per_class: int | None = None,
        augmenter: Augmenter | None = None,
        warmup: int = -1,
    ):
        super().__init__(
            x,
            y,
            load_example_fn=load_example_fn,
            classes_per_batch=classes_per_batch,
            examples_per_class_per_batch=examples_per_class_per_batch,
            steps_per_epoch=steps_per_epoch,
            class_list=class_list,
            total_examples_per_class=total_examples_per_class,
            augmenter=augmenter,
            warmup=warmup,
        )
