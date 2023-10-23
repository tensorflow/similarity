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
"""Efficiently find nearest indexed embeddings

The search module is used to find the closest indexed example embeddings
to a query example embebbeding.
To do so it performs a sub-linear time
[ANN (Approximate Nearst Neigboors)](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
search on the indexed set of embedding examples.

Different ANN librairies have [different performance profiles](https://github.com/erikbern/ann-benchmarks).
Tensorflow Similarity by default use [NMSLIB](https://github.com/nmslib/nmslib)
which has a strong performance profile and is very portable.

Adding another backend is fairly straightforward: subclass the
abstract class `Search()` and implement the abstract methods. Then to use it
pass it to the `compile()` method of your [SimilarityModel].

Adding your search backend as a built-in choice invlolves
modifiying the [Indexer](../indexer.md) and sending a PR. In general, unless
the backend is of general use, its better to not include it as
a built-in option as it must be supported moving forward.
"""
from __future__ import annotations

from typing import Any, Type

try:
    from tensorflow.keras.utils import deserialize_keras_object, serialize_keras_object
except ImportError:
    from tensorflow.keras.utils.legacy import (
        deserialize_keras_object,
        serialize_keras_object,
    )

from .faiss import FaissSearch  # noqa
from .linear import LinearSearch  # noqa
from .nmslib import NMSLibSearch  # noqa
from .search import Search  # noqa


def serialize(search: Search) -> dict[str, Any]:
    """Serialize the search configuration to JSON compatible python dict.

    The configuration can be used for persistence and reconstruct the `Search`
    instance again.

    >>> tfsim.search.serialize(tfsim.search.LinearSearch())
    {'class_name': 'LinearSearch', 'config': { 'name': 'LinearSearch',
                                     'distance': 'cosine',
                                     'dim': 10, 'verbose': 0}}

    Args:
      search: A `Search` instance to serialize.

    Returns:
      Python dict which contains the configuration of the search.
    """
    config: dict[str, Any] = serialize_keras_object(search)
    return config


def deserialize(config, custom_objects=None) -> Search:
    """Inverse of the `serialize` function.

    Args:
        config: search configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
          objects (classes and functions) to be considered during deserialization.

    Returns:
        A search instance.
    """
    all_classes: dict[str, Type[Search]] = {
        "faiss": FaissSearch,
        "faisssearch": FaissSearch,
        "linear": LinearSearch,
        "linearsearch": LinearSearch,
        "nmslib": NMSLibSearch,
        "nmslibsearch": NMSLibSearch,
    }

    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in all_classes:
        config["class_name"] = config["class_name"].lower()
    search: Search = deserialize_keras_object(
        config, module_objects=all_classes, custom_objects=custom_objects, printable_module_name="search"
    )
    return search


def get(identifier, **kwargs) -> Search:
    """Retrieves a search instance.

    Args:
        identifier: search identifier, one of
            - String: name of a search class.
            - Dictionary: configuration dictionary.
            - search instance (it will be returned unchanged).
        **kwargs: Additional keyword arguments to be passed to the Search
            constructor. Used as the config if `identifier` is a str.

    Returns:
        A search instance.

    Raises:
        ValueError: If `identifier` cannot be interpreted.
    """
    if isinstance(identifier, Search):
        return identifier
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": kwargs}
        return deserialize(config)
    else:
        raise ValueError("Could not interpret search identifier: {}".format(identifier))
