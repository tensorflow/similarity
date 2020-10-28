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

from tensorflow_similarity.api.engine.filter import MoiraiPairFilter
from tensorflow_similarity.utils.config_utils import register_custom_object
import numpy as np


def edit_distance(left, right):
    len_l = len(left) + 1
    len_r = len(right) + 1

    table = {}
    for i in range(len_l):
        table[i, 0] = i
    for j in range(len_r):
        table[0, j] = j

    for i in range(1, len_l):
        for j in range(1, len_r):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            table[i, j] = min(table[i, j - 1] + 1, table[i - 1, j] + 1,
                              table[i - 1, j - 1] + cost)
    return table[i, j]


class EditDistanceFilter(MoiraiPairFilter):
    def __init__(self,
                 feature_name=None,
                 min_length=5,
                 min_distance=4,
                 max_distance=10,
                 raw=True):
        super(EditDistanceFilter, self).__init__()
        self.feature_name = feature_name
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_length = min_length
        self.raw = raw

    def keep_positive(self, a, b):
        return True

    def keep_negative(self, a, b):
        if self.raw:
            a_v = a.raw_feature_dictionary[self.feature_name]
            b_v = b.raw_feature_dictionary[self.feature_name]
        else:
            a_v = a.feature_dictionary[self.feature_name]
            b_v = b.feature_dictionary[self.feature_name]

        if len(a_v) < self.min_length or len(b_v) < self.min_length:
            return False

        distance = edit_distance(a_v, b_v)
        return (distance >= self.min_distance) and (distance <=
                                                    self.max_distance)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'feature_name': self.feature_name,
                'min_distance': self.min_distance,
                'max_distance': self.max_distance,
                'min_length': self.min_length,
                'raw': self.raw
            }
        }


register_custom_object("EditDistanceFilter", EditDistanceFilter)
