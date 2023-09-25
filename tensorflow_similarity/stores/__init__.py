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
""" Key Values Stores store the data associated with the embeddings indexed by
the `Indexer()`.

Each key of the store represent a **record** that contains information
about a given embedding.

The main use-case for the store is to retrieve the records associated
with the ids returned by a nearest neigboor search performed with the
[`Search()` module](../search/).

Additionally one might want to inspect the content of the index which is why
`Store()` class may implement an export to
a [Pandas Dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
via the `to_pandas()` method.
"""
from __future__ import annotations

from typing import Any, Type

from tensorflow.python.keras.utils.generic_utils import (
    deserialize_keras_object,
    serialize_keras_object,
)

from .cached import CachedStore  # noqa
from .memory import MemoryStore  # noqa
from .redis import RedisStore  # noqa
from .store import Store  # noqa


def serialize(store: Store) -> dict[str, Any]:
    """Serialize the store configuration to JSON compatible python dict.

    The configuration can be used for persistence and reconstruct the `Store`
    instance again.

    >>> tfsim.stores.serialize(tfsim.stores.CachedStore())
    {'class_name': 'CachedStore', 'config': { 'name': 'CachedStore',
                                     'shared_size': 10, 'path': '.',
                                     'num_items': 0, 'verbose': 0}}

    Args:
      store: A `Store` instance to serialize.

    Returns:
      Python dict which contains the configuration of the store.
    """
    config: dict[str, Any] = serialize_keras_object(store)
    return config


def deserialize(config, custom_objects=None) -> Store:
    """Inverse of the `serialize` function.

    Args:
        config: store configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
          objects (classes and functions) to be considered during deserialization.

    Returns:
        A store instance.
    """
    all_classes: dict[str, Type[Store]] = {
        "cached": CachedStore,
        "cachedstore": CachedStore,
        "memory": MemoryStore,
        "memorystore": MemoryStore,
        "redis": RedisStore,
        "redisstore": RedisStore,
    }

    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in all_classes:
        config["class_name"] = config["class_name"].lower()
    store: Store = deserialize_keras_object(
        config, module_objects=all_classes, custom_objects=custom_objects, printable_module_name="stores"
    )
    return store


def get(identifier) -> Store:
    """Retrieves a store instance.

    Args:
        identifier: store identifier, one of
            - String: name of a store class.
            - Dictionary: configuration dictionary.
            - store instance (it will be returned unchanged).

    Returns:
        A store instance.

    Raises:
        ValueError: If `identifier` cannot be interpreted.
    """
    if isinstance(identifier, Store):
        return identifier
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        return deserialize(config)
    else:
        raise ValueError("Could not interpret Store identifier: {}".format(identifier))
