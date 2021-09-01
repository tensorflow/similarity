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

from .store import Store  # noqa
from .memory_store import MemoryStore  # noqa
