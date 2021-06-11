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

import dataclasses
from typing import Any, Callable, Iterable, Union, Optional, List, Tuple, Optional  # noqa

import numpy as np
import tensorflow as tf


class PandasDataFrame(object):
    """Symbolic pandas frame
    Pandas type are too loose you get an Any. We want a PandaFrame
    """
    pass


class Tensor(List):
    """The base class of all dense Tensor objects.

    A dense tensor has a static data type (dtype), and may have a static rank
    and shape. Tensor objects are immutable. Mutable objects may be backed by
    a Tensor which holds the unique handle that identifies the mutable object.
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


class BoolTensor(Tensor):
    """Bool tensor """


class FloatTensor(Tensor):
    """Float tensor """


class IntTensor(Tensor):
    """Integer tensor"""


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


EqFunc = Callable[[Any, Any], bool]


def _optional_eq(a: Any, b: Any, eq_fun: EqFunc) -> bool:
    """__eq__ for Optional[Any] types."""
    if a is None:
        return b is None
    elif b is None:
        return False

    return eq_fun(a, b)


def _basic_eq(a: Any, b: Any) -> bool:
    eq: bool = a == b
    return eq


def _ndarray_eq(a: np.ndarray, b: np.ndarray) -> bool:
    eq: bool = np.allclose(a, b, rtol=0, atol=0, equal_nan=True)
    return eq


def _tf_eq(a: Tensor, b: Tensor) -> bool:
    eq: bool = tf.math.reduce_all(tf.math.equal(a, b))
    return eq


@dataclasses.dataclass
class Lookup:
    """Metadata associated with a query match.

    Attributes:
        rank: Rank of the match with respect to the query distance.

        distance: The distance from the match to the query.

        label: The label associated with the match. Default None.

        embedding: The embedded match vector. Default None.

        data: The original Tensor representation of the match result.
        Default None.
    """
    rank: int
    distance: float
    label: Optional[int] = dataclasses.field(default=None)
    embedding: Optional[np.ndarray] = dataclasses.field(default=None)
    data: Optional[Tensor] = dataclasses.field(default=None)

    def __eq__(self, other) -> bool:
        if other.__class__ is not self.__class__:
            return False
        if self.rank != other.rank:
            return False
        if self.distance != other.distance:
            return False
        if not _optional_eq(self.label, other.label, _basic_eq):
            return False
        if not _optional_eq(self.embedding, other.embedding, _ndarray_eq):
            return False
        if not _optional_eq(self.data, other.data, _tf_eq):
            return False

        return True
