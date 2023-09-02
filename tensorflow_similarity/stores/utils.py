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

from typing import Any, Callable, Dict

from .store import Store

LoadFn = Callable[[Dict[str, Any]], Store]


def load_redis(config: Dict[str, Any]):
    from .redis import RedisStore

    return RedisStore(**config)


def load_cached(config: Dict[str, Any]):
    from .cached import CachedStore

    return CachedStore(**config)


def load_memory(config: Dict[str, Any]):
    from .memory import MemoryStore

    return MemoryStore(**config)


STORE_ALIASES: Dict[str, LoadFn] = {
    "redisstore": load_redis,
    "cachedstore": load_cached,
    "memorystore": load_memory,
    "redis": load_redis,
    "cached": load_cached,
    "memory": load_memory,
}


def make_store(config: Dict[str, Any]) -> Store:
    """Creates a store instance from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same search from the config

    Args:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        A Store instance.
    """

    canonical_name = config["canonical_name"].lower()
    if canonical_name in STORE_ALIASES:
        store: Store = STORE_ALIASES[canonical_name](config)
    else:
        raise ValueError(f"Unknown search type: {canonical_name}")

    return store
