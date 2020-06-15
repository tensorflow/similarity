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

from tensorflow_similarity.api.generators.tuples import QuadrupletGenerator
import numpy as np


def gen_testdata():
    intinput = np.random.rand(10, 2)
    intinputv = np.random.rand(10, 2) + 2.0
    x = {"intinput": intinput, "intinputv": intinputv}
    y = np.random.randint(0, 10, size=10)

    return x, y


def test_task_based_generator():
    x, y = gen_testdata()
    generator = QuadrupletGenerator(x, y)
    batch = generator.get_batch(0)

    assert np.array_equal(batch.labels["anchor_idx_out"].shape, (128,))
    assert np.array_equal(batch.labels["pos_idx_out"].shape, (128,))
    assert np.array_equal(batch.labels["neg1_idx_out"].shape, (128,))
    assert np.array_equal(batch.labels["neg2_idx_out"].shape, (128,))

    assert np.array_equal(batch.raw_values["anchor_intinput"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["anchor_intinputv"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["pos_intinput"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["pos_intinputv"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["neg1_intinput"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["neg1_intinputv"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["neg2_intinput"].shape, (128, 2))
    assert np.array_equal(batch.raw_values["neg2_intinputv"].shape, (128, 2))

    assert np.array_equal(batch.values["anchor_intinput"].shape, (128, 2))
    assert np.array_equal(batch.values["anchor_intinputv"].shape, (128, 2))
    assert np.array_equal(batch.values["pos_intinput"].shape, (128, 2))
    assert np.array_equal(batch.values["pos_intinputv"].shape, (128, 2))
    assert np.array_equal(batch.values["neg1_intinput"].shape, (128, 2))
    assert np.array_equal(batch.values["neg1_intinputv"].shape, (128, 2))
    assert np.array_equal(batch.values["neg2_intinput"].shape, (128, 2))
    assert np.array_equal(batch.values["neg2_intinputv"].shape, (128, 2))
