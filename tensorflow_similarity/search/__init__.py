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
from .nmslib_search import NMSLibSearch # noqa
from .search import Search  # noqa
