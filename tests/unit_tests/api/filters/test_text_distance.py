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

from tensorflow_similarity.api.engine.generator import Element
from tensorflow_similarity.api.filters.text_distance import EditDistanceFilter

import numpy as np


def example(name):
    return Element(index=0,
                   label_id=0,
                   augmented=0,
                   feature_dictionary={"examples": name},
                   raw_feature_dictionary={},
                   preprocessed_dictionary={})


def raw_example(name):
    return Element(index=0,
                   label_id=0,
                   augmented=0,
                   raw_feature_dictionary={"examples": name},
                   feature_dictionary={},
                   preprocessed_dictionary={})


def test_permissive_settings():
    distance_filter = EditDistanceFilter(feature_name='examples',
                                         min_distance=0,
                                         max_distance=1000,
                                         min_length=0,
                                         raw=False)

    llama = example("llama")
    other = example("googlegooglegoogle")

    assert True == distance_filter.keep_negative(llama, other)


def test_permissive_settings_raw():
    distance_filter = EditDistanceFilter(feature_name='examples',
                                         min_distance=0,
                                         max_distance=1000,
                                         min_length=0,
                                         raw=True)

    llama = raw_example("llama")
    other = raw_example("googlegooglegoogle")

    assert True == distance_filter.keep_negative(llama, other)


def test_min_settings():
    distance_2_filter = EditDistanceFilter(feature_name='examples',
                                           min_distance=2,
                                           max_distance=1000,
                                           min_length=0,
                                           raw=False)

    distance_1_filter = EditDistanceFilter(feature_name='examples',
                                           min_distance=1,
                                           max_distance=1000,
                                           min_length=0,
                                           raw=False)

    llama = example("google")
    other = example("googlf")

    assert False == distance_2_filter.keep_negative(llama, other)
    assert True == distance_1_filter.keep_negative(llama, other)


def test_min_settings_raw():
    distance_2_filter = EditDistanceFilter(feature_name='examples',
                                           min_distance=2,
                                           max_distance=1000,
                                           min_length=0,
                                           raw=True)

    distance_1_filter = EditDistanceFilter(feature_name='examples',
                                           min_distance=1,
                                           max_distance=1000,
                                           min_length=0,
                                           raw=True)

    llama = raw_example("google")
    other = raw_example("googlf")

    assert False == distance_2_filter.keep_negative(llama, other)
    assert True == distance_1_filter.keep_negative(llama, other)


def test_max_settings():
    distance_2_filter = EditDistanceFilter(feature_name='examples',
                                           min_distance=0,
                                           max_distance=2,
                                           min_length=0,
                                           raw=False)

    distance_1_filter = EditDistanceFilter(feature_name='examples',
                                           min_distance=0,
                                           max_distance=1,
                                           min_length=0,
                                           raw=False)

    llama = example("google")
    other = example("googff")

    assert True == distance_2_filter.keep_negative(llama, other)
    assert False == distance_1_filter.keep_negative(llama, other)
