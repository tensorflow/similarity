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

"""Distance functions.

This module provides tensor based distance functions for computing distances
between batches of tensors.

Available distance functions:
  l1 - L1 distance between the tensors.
  l2 - L2 distance between the tensors.
"""

from tensorflow.keras import backend as K
from tensorflow_similarity.utils.config_utils import register_custom_object
import numpy as np


# Distance computations on Keras tensors.
def l1(left, right):
    return K.sum(K.abs(left - right), axis=-1, keepdims=False)


def l2(left, right):
    return K.sqrt(
        K.maximum(
            K.sum(K.square(left - right), axis=1, keepdims=False),
            K.epsilon()))


def get_distance_by_name(str):
    fns = {'l1': l1, 'l2': l2}
    return fns[str]


# Eager computations on np arrays.
def l1_eager(left, right):
    return np.linalg.norm(left, right, ord=1)


def l2_eager(left, right):
    return np.linalg.norm(left, right, ord=2)


def get_eager_distance_by_name(str):
    fns = {'l1': l1_eager, 'l2': l2_eager}
    return fns[str]


class L1(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, left, right):
        return K.sum(K.abs(left - right), axis=-1)

    def get_config(self):
        return {'class_name': self.__class__.__name__, 'config': {}}


register_custom_object("L1", L1)


class L2(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, left, right):
        return K.sqrt(
            K.maximum(
                K.sum(K.square(left - right), axis=1, keepdims=True),
                K.epsilon()))

    def get_config(self):
        return {'class_name': self.__class__.__name__, 'config': {}}


register_custom_object("L2", L2)
