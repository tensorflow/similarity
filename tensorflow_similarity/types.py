# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core TensorFlow types."""

from typing import Iterable, Union, Optional, List, Tuple, Optional  # noqa
import numpy as np


class PandasDataFrame(object):
    """Symbolic pandas frame
    Pandas type are too loose you get an Any. We want a PandaFrame
  """


class Tensor(List):
    """The base class of all dense Tensor objects.

  A dense tensor has a static data type (dtype), and may have a static rank and
  shape. Tensor objects are immutable. Mutable objects may be backed by a
  Tensor which holds the unique handle that identifies the mutable object.
  """
    @property
    def dtype(self):
        pass

    @property
    def shape(self):
        pass

    @property
    def __len__(self):
        pass

    @property
    def __iter__(self):
        pass


class IntTensor(Tensor):
    """Integer tensor"""


class FloatTensor(Tensor):
    """Float tensor """


class Symbol(Tensor):
    """Symbolic "graph" Tensor.

  These objects represent the output of an op definition and do not carry a
  value.
  """
    pass


class Value(Tensor):
    """Tensor that can be associated with a value (aka "eager tensor").

  These objects represent the (usually future) output of executing an op
  immediately.
  """
    def numpy(self):
        pass

