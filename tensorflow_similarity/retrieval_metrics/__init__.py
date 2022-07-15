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
"""Retrieval metrics measure the quality of the embedding space given a
set query examples and a set of indexed examples. Informally it can be thought
of as how well the space is clustered among other things.
"""
from .bndcg import BNDCG  # noqa
from .map_at_k import MapAtK  # noqa
from .precision_at_k import PrecisionAtK  # noqa
from .recall_at_k import RecallAtK  # noqa
from .retrieval_metric import RetrievalMetric  # noqa
