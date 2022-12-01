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

import tensorflow as tf

from tensorflow_similarity.samplers.samplers import Sampler


def batch_class_ratio(sampler: Sampler, num_batches: int = 100) -> float:
    """Computes the average number of examples per class within each batch.
    Similarity learning requires at least 2 examples per class in each batch.
    This is needed in order to construct the triplets. This function
    provides the average number of examples per class within each batch and
    can be used to check that a sampler is working correctly.
    The ratio should be >= 2.
    Args:
        sampler: A tf.similarity sampler object.
        num_batches: The number of batches to sample.
    Returns:
        The average number of examples per class.
    """
    ratio = 0
    for batch_count, (_, y) in enumerate(sampler):
        if batch_count < num_batches:
            batch_size = tf.shape(y)[0]
            num_classes = tf.shape(tf.unique(y)[0])[0]
            ratio += tf.math.divide(batch_size, num_classes)
        else:
            break

    return float(ratio / (batch_count + 1))
