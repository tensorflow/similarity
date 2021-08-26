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
    return tf.cast(tf.constant(all_values), dtype='int32')


def unpack_lookup_distances(
        lookups: Sequence[Sequence[Lookup]],
        distance_rounding: Optional[int] = None) -> FloatTensor:
    # using list comprehension as it is faster
    all_values = [[n.distance for n in lu] for lu in lookups]
    dists = tf.cast(tf.constant(all_values), dtype='float32')

    if distance_rounding is not None:
        multiplier = 10.0**distance_rounding
        dists = tf.round(dists * multiplier) / multiplier
    return dists
