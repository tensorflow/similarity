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
"""Vectorized embedding pairwise distances computation functions"""
from __future__ import annotations

from typing import Any, Type

from tensorflow.python.keras.utils.generic_utils import (
    deserialize_keras_object,
    serialize_keras_object,
)

from .cosine import CosineDistance  # noqa
from .distance import Distance  # noqa
from .euclidean import EuclideanDistance  # noqa
from .euclidean import SquaredEuclideanDistance  # noqa
from .inner_product import InnerProductSimilarity  # noqa
from .manhattan import ManhattanDistance  # noqa
from .snr import SNRDistance  # noqa

_ALL_CLASSES: dict[str, Type[Distance]] = {
    "inner_product": InnerProductSimilarity,
    "innerproductsimilarity": InnerProductSimilarity,
    "ip": InnerProductSimilarity,
    "cosine": CosineDistance,
    "cosinedistance": CosineDistance,
    "euclidean": EuclideanDistance,
    "euclideandistance": EuclideanDistance,
    "l2": EuclideanDistance,
    "pythagorean": EuclideanDistance,
    "squared_euclidean": SquaredEuclideanDistance,
    "squaredeuclideandistance": SquaredEuclideanDistance,
    "sql2": SquaredEuclideanDistance,
    "sqeuclidean": SquaredEuclideanDistance,
    "manhattan": ManhattanDistance,
    "manhattandistance": ManhattanDistance,
    "l1": ManhattanDistance,
    "taxicab": ManhattanDistance,
    "snr": SNRDistance,
    "snrdistance": SNRDistance,
    "signal-to-noise-ratio": SNRDistance,
}


def serialize(distance: Distance) -> dict[str, Any]:
    """Serialize the distance configuration to JSON compatible python dict.

    The configuration can be used for persistence and reconstruct the `Distance`
    instance again.

    >>> tfsim.distances.serialize(tfsim.distances.CosineDistance())
    {'class_name': 'CosineDistance', 'config': { 'name': 'CosineDistance'}}

    Args:
      search: A `Distance` instance to serialize.

    Returns:
      Python dict which contains the configuration of the distance.
    """
    config: dict[str, Any] = serialize_keras_object(distance)
    return config


def deserialize(config, custom_objects=None) -> Distance:
    """Inverse of the `serialize` function.

    Args:
        config: distance configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
          objects (classes and functions) to be considered during deserialization.

    Returns:
        A distance instance.
    """
    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in _ALL_CLASSES:
        config["class_name"] = config["class_name"].lower()
    distance: Distance = deserialize_keras_object(
        config, module_objects=_ALL_CLASSES, custom_objects=custom_objects, printable_module_name="distances"
    )
    return distance


def get(identifier) -> Distance:
    """Retrieves a distance instance.

    Args:
        identifier: distance identifier, one of
            - String: name of a distance class.
            - Dictionary: configuration dictionary.
            - distance instance (it will be returned unchanged).

    Returns:
        A distance instance.

    Raises:
        ValueError: If `identifier` cannot be interpreted.
    """
    if isinstance(identifier, Distance):
        return identifier
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(config)
    else:
        raise ValueError("Could not interpret search identifier: {}".format(identifier))
