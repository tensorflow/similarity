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

from collections.abc import Callable
from typing import Any, Dict

from .search import Search

LoadFn = Callable[[Dict[str, Any]], Search]


def load_faiss(config: Dict[str, Any]):
    from .faiss import Faiss

    return Faiss(**config)


def load_linear(config: Dict[str, Any]):
    from .linear import Linear

    return Linear(**config)


def load_nmslib(config: Dict[str, Any]):
    from .nmslib import NMSLib

    return NMSLib(**config)


SEARCH_ALIASES: Dict[str, LoadFn] = {
    "faiss": load_faiss,
    "linear": load_linear,
    "nmslib": load_nmslib,
}


def make_search(config: Dict[str, Any]) -> Search:
    """Creates a search instance from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same search from the config

    Args:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        A search instance.
    """

    canonical_name = config["canonical_name"].lower()
    if canonical_name in SEARCH_ALIASES:
        search: Search = SEARCH_ALIASES[canonical_name](config)
    else:
        raise ValueError(f"Unknown search type: {canonical_name}")

    return search
