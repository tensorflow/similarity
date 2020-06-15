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

import numpy as np
from tensorflow_similarity.readers.concatenated_view import ConcatenatedView
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


TEST_DATA = [
    np.ones([3, 2, 2]) * 5,
    np.ones([5, 2, 2]) * 2,
    np.ones([1, 2, 2]),
    np.ones([4, 2, 2]) * 4
]
COMBINED_TEST_DATA = np.concatenate(TEST_DATA)


def test_matrix_construction():
    m = ConcatenatedView(TEST_DATA)
    assert len(m) == 13


def test_single_shard_slices():
    m = ConcatenatedView(TEST_DATA)
    assert np.allclose(m[0:3], TEST_DATA[0])
    assert np.allclose(m[3:8], TEST_DATA[1])
    assert np.allclose(m[8:9], TEST_DATA[2])
    assert np.allclose(m[9:13], TEST_DATA[3])


def test_two_shard_slices():
    m = ConcatenatedView(TEST_DATA)
    assert np.allclose(m[0:8], np.concatenate([TEST_DATA[0], TEST_DATA[1]]))
    assert np.allclose(m[3:9], np.concatenate([TEST_DATA[1], TEST_DATA[2]]))
    assert np.allclose(m[8:13], np.concatenate([TEST_DATA[2], TEST_DATA[3]]))


def test_many_shard_slices():
    m = ConcatenatedView(TEST_DATA)
    for i in range(len(TEST_DATA)):
        for j in range(i, len(TEST_DATA)):
            assert np.allclose(m[i:j], COMBINED_TEST_DATA[i:j])


def test_lookups():
    m = ConcatenatedView(TEST_DATA)
    for i in range(len(TEST_DATA)):
        assert np.allclose(m[i], COMBINED_TEST_DATA[i])


def test_many_lookups():
    m = ConcatenatedView(TEST_DATA)
    for i in range(len(TEST_DATA)):
        for tests in range(10):
            idxs = np.random.choice(len(TEST_DATA), size=i)
            assert np.allclose(m[idxs],
                               np.take(COMBINED_TEST_DATA, idxs, axis=0))
