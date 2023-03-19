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
from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor, Lookup


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


def unpack_lookup_labels(lookups: Sequence[Sequence[Lookup]], dtype: str | tf.DType) -> IntTensor:
    # Using list comprehension as it is faster
    all_values = [[n.label for n in lu] for lu in lookups]
    # Lookup sets are not guaranteed to all be the same size. Therefore we load
    # the list of lists as a ragged tensor and convert to an int tensor with a
    # default class label value set to the max value for int.
    base_type = tf.dtypes.as_dtype(dtype).base_dtype
    ragged_labels = tf.ragged.constant(all_values, dtype=base_type)

    if not _same_length_rows(ragged_labels):
        print(
            f"WARNING: {_count_of_small_lookup_sets(ragged_labels)} lookup "
            "sets are shorter than the max lookup set length. Imputing "
            "0x7FFFFFFF for the missing label lookups."
        )

    result: IntTensor = ragged_labels.to_tensor(default_value=0x7FFFFFFF)

    return result


def unpack_lookup_distances(
    lookups: Sequence[Sequence[Lookup]], dtype: str | tf.DType, distance_rounding: int | None = None
) -> FloatTensor:
    # using list comprehension as it is faster
    all_values = [[n.distance for n in lu] for lu in lookups]
    # Lookup sets are not guaranteed to all be the same size. Therefore we load
    # the list of lists as a ragged tensor and convert to an flat32 tensor with a
    # default dist value set to math.inf.
    base_type = tf.dtypes.as_dtype(dtype).base_dtype
    ragged_dists = tf.ragged.constant(all_values, dtype=base_type)

    if distance_rounding is not None:
        multiplier = tf.constant([10.0**distance_rounding])
        ragged_dists = tf.round(ragged_dists * multiplier) / multiplier

    if not _same_length_rows(ragged_dists):
        print(
            f"WARNING: {_count_of_small_lookup_sets(ragged_dists)} lookup "
            "sets are shorter than the max lookup set length. Imputing "
            "math.inf for the missing distance lookups."
        )

    dists: FloatTensor = ragged_dists.to_tensor(default_value=math.inf)

    return dists


def unpack_results(
    results: Mapping[str, np.ndarray],
    epoch: int,
    logs: MutableMapping[str, Any],
    tb_writer: tf.summary.SummaryWriter,
    name_suffix: str | None = "",
) -> list[str]:
    """Updates logs, writes summary, and returns list of strings of
    evaluation metric"""
    mstr = []
    for metric_name, vals in results.items():
        float_val = vals[0] if isinstance(vals, np.ndarray) else vals
        full_metric_name = f"{metric_name}{name_suffix}"
        logs[full_metric_name] = float_val
        mstr.append(f"{full_metric_name}: {float_val:.4f}")
        if tb_writer:
            with tb_writer.as_default():
                tf.summary.scalar(full_metric_name, float_val, step=epoch)
    return mstr


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
