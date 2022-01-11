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
import math
from typing import Optional, Sequence

import tensorflow as tf

from tensorflow_similarity.types import BoolTensor
from tensorflow_similarity.types import FloatTensor
from tensorflow_similarity.types import IntTensor
from tensorflow_similarity.types import Lookup


def is_tensor_or_variable(x):
    "check if a variable is tf.Tensor or tf.Variable"
    return tf.is_tensor(x) or isinstance(x, tf.Variable)


def tf_cap_memory():
    "Avoid TF to hog memory before needing it"
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def unpack_lookup_labels(lookups: Sequence[Sequence[Lookup]]) -> IntTensor:
    # Using list comprehension as it is faster
    all_values = [[n.label for n in lu] for lu in lookups]
    # Lookup sets are not guaranteed to all be the same size. Therefore we load
    # the list of lists as a ragged tensor and convert to an int32 tensor with a
    # default class label value set to the max value for int32.
    ragged_labels = tf.ragged.constant(all_values, dtype="int32")

    if not _same_length_rows(ragged_labels):
        print(f"WARNING: {_count_of_small_lookup_sets(ragged_labels)} lookup "
              "sets are shorter than the max lookup set length. Imputing "
              "0x7FFFFFFF for the missing label lookups.")

    result: IntTensor = ragged_labels.to_tensor(default_value=0x7FFFFFFF)

    return result


def unpack_lookup_distances(
        lookups: Sequence[Sequence[Lookup]],
        distance_rounding: Optional[int] = None) -> FloatTensor:
    # using list comprehension as it is faster
    all_values = [[n.distance for n in lu] for lu in lookups]
    # Lookup sets are not guaranteed to all be the same size. Therefore we load
    # the list of lists as a ragged tensor and convert to an flat32 tensor with a
    # default dist value set to math.inf.
    ragged_dists = tf.ragged.constant(all_values, dtype="float32")

    if distance_rounding is not None:
        multiplier = tf.constant([10.0**distance_rounding])
        ragged_dists = tf.round(ragged_dists * multiplier) / multiplier

    if not _same_length_rows(ragged_dists):
        print(f"WARNING: {_count_of_small_lookup_sets(ragged_dists)} lookup "
              "sets are shorter than the max lookup set length. Imputing "
              "math.inf for the missing distance lookups.")

    dists: FloatTensor = ragged_dists.to_tensor(default_value=math.inf)

    return dists


def _same_length_rows(x: tf.RaggedTensor) -> BoolTensor:
    """Check if the rows are all the same length.

    Args:
        x: A RaggedTensor

    Returns:
       True if all rows are the same length.
    """
    dims = tf.expand_dims(x.row_lengths(), axis=-1)
    pairwise_equality = tf.equal(dims, tf.transpose(dims))
    is_same_length: BoolTensor = tf.math.reduce_all(pairwise_equality)
    return is_same_length


def _count_of_small_lookup_sets(x: tf.RaggedTensor) -> IntTensor:
    """The count of lookup sets smaller than x.bounding_shape()[1]

    Args:
        x: A RaggedTensor

    Returns:
        The count of smaller lookup sets.
    """
    rl = x.row_lengths()
    max_rl = x.bounding_shape()[1]
    short_lookup_sets = tf.cast(rl != max_rl, dtype="int32")
    small_lookup_count: IntTensor = tf.math.reduce_sum(short_lookup_sets)
    return small_lookup_count
