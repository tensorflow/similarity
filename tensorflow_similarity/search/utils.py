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

from typing import Any, Type

from .nmslib_search import NMSLibSearch
from .search import Search

SEARCH_ALIASES: dict[str, Type[Search]] = {
    "NMSLibSearch": NMSLibSearch,
}


def make_search(config: dict[str, Any]) -> Search:
    """Creates a search instance from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same search from the config

    Args:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        A search instance.
    """

    if config["canonical_name"] in SEARCH_ALIASES:
        search: Search = SEARCH_ALIASES[config["canonical_name"]](**config)
    else:
        raise ValueError(f"Unknown search type: {config['canonical_name']}")

    return search
