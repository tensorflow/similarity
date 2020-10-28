# Copyright 2020 Google LLC
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

from __future__ import absolute_import

__all__ = ['base', 'hdf5', 'text', 'keras_datasets', 'inmemory']

from . import base
from . import hdf5
from . import keras_datasets
from . import inmemory

from .base import Reader
from .hdf5 import H5Reader, InMemoryH5Reader
from .inmemory import MemoryReader
from .text import TextReader
from .keras_datasets import KerasDatasetReader
