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
"Various utilities functions for improved quality of life."
from typing import Optional, Sequence

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor
from tensorflow_similarity.types import IntTensor
from tensorflow_similarity.types import Lookup


def is_tensor_or_variable(x):
    "check if a variable is tf.Tensor or tf.Variable"
    return tf.is_tensor(x) or isinstance(x, tf.Variable)


def tf_cap_memory():
    "Avoid TF to hog memory before needing it"
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def unpack_lookup_labels(lookups: Sequence[Sequence[Lookup]]) -> IntTensor:
    # using list comprehension as it is faster
    all_values = [[n.label for n in lu] for lu in lookups]
    result: IntTensor = tf.cast(tf.constant(all_values), dtype='int32')
    return result


def unpack_lookup_distances(
        lookups: Sequence[Sequence[Lookup]],
        distance_rounding: Optional[int] = None) -> FloatTensor:
    # using list comprehension as it is faster
    all_values = [[n.distance for n in lu] for lu in lookups]
    dists: FloatTensor = tf.cast(tf.constant(all_values), dtype='float32')

    if distance_rounding is not None:
        multiplier = tf.constant([10.0**distance_rounding])
        dists = tf.round(dists * multiplier) / multiplier

    return dists
