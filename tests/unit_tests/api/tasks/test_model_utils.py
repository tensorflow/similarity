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

from tensorflow_similarity.utils.model_utils import training_data_to_example_list, example_list_to_training_data
import collections
import six


def test_training_data_to_example_list():
    inp = {
        "example_1": [1, 2, 3],
        "example_2": [4, 5, 6],
    }

    out = training_data_to_example_list(inp)

    assert len(out) == 3
    assert out[0]["example_1"] == 1
    assert out[1]["example_1"] == 2
    assert out[2]["example_1"] == 3

    assert out[0]["example_2"] == 4
    assert out[1]["example_2"] == 5
    assert out[2]["example_2"] == 6


def test_example_list_to_training_data():
    inp = [
        {
            "example_1": 1,
            "example_2": 4,
        },
        {
            "example_1": 2,
            "example_2": 5,
        },
        {
            "example_1": 3,
            "example_2": 6,
        }
    ]

    out = example_list_to_training_data(inp)

    assert len(out) == 2
    assert out["example_1"][0] == 1
    assert out["example_1"][1] == 2
    assert out["example_1"][2] == 3

    assert out["example_2"][0] == 4
    assert out["example_2"][1] == 5
    assert out["example_2"][2] == 6
