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

import collections
import tensorflow_similarity
from tensorflow_similarity.api.generators.tuples import *
import numpy as np


def _ex(f1, f2):
    return {"example_f1": f1, "example_f2": f2}


def simple_test_set():

    x = {
        "example_f1": [],
        "example_f2": [],
    }

    y = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

    for i in y:
        x["example_f1"].append(3.0 * i)
        x["example_f2"].append(5.0 * i)
    return x, y


def preprocessing(x):
    return {"example_f1": x["example_f1"], "example_f2": x["example_f2"]}


def augmentation(x):
    return {
        "example_f1": x["example_f1"] - .0625,
        "example_f2": x["example_f2"] - .125,
    }


def test_quad_gen_construction():
    x, y = simple_test_set()
    qg = QuadrupletGenerator(x,
                             y,
                             preprocessing=preprocessing,
                             augmentation=augmentation)


def test_small_random_sample():
    x, y = simple_test_set()
    qg = QuadrupletGenerator(x,
                             y,
                             preprocessing=preprocessing,
                             augmentation=augmentation)

    counts = [0] * 5
    for i in range(10000):
        res = qg.small_random_sample(5, 3)
        for r in res:
            counts[r] += 1

    np.random.seed(3713)
    expected = [6000, 6000, 6000, 6000, 6000]

    np.allclose(expected, counts, 100)


def test_quad_get_batch():
    x, y = simple_test_set()
    qg = QuadrupletGenerator(x,
                             y,
                             preprocessing=preprocessing,
                             augmentation=augmentation,
                             batch_size=10240)

    batch = qg.get_batch(0)

    anc = batch.values["anchor_example_f1"]
    neg = batch.values["neg1_example_f1"]
    neg2 = batch.values["neg2_example_f1"]
    pos = batch.values["pos_example_f1"]

    assert len(anc) == 10240
    assert len(pos) == 10240
    assert len(neg) == 10240
    assert len(neg2) == 10240

    # In our simple test, all items for the same class have the same value,
    # +/- .0625
    np.allclose(pos, anc, atol=.063)

    # None of the other pairs of elements should be the same, since their classes
    # are different, and thus should have different values.
    overall_count = collections.defaultdict(int)
    a_count = collections.defaultdict(int)
    n1_count = collections.defaultdict(int)
    n2_count = collections.defaultdict(int)
    pos_count = collections.defaultdict(int)

    for a, n1, n2, pos in zip(anc, neg, neg2, pos):
        assert n1 != n2, "Bad tuple n1==n2: %s %s %s %s" % ((a, n1, n2, pos))
        assert n1 != pos, "Bad tuple n1==pos: %s %s %s %s" % ((a, n1, n2, pos))
        assert n2 != pos, "Bad tuple n2==pos: %s %s %s %s" % ((a, n1, n2, pos))
        assert n1 != a, "Bad tuple n1==a: %s %s %s %s" % ((a, n1, n2, pos))
        assert n2 != a, "Bad tuple n2==a: %s %s %s %s" % ((a, n1, n2, pos))
        assert a == np.round(pos), "Bad tuple a!=pos: %s %s %s %s" % (
            (a, n1, n2, pos))

        overall_count[a] += 1
        a_count[a] += 1

        overall_count[pos] += 1
        pos_count[pos] += 1

        overall_count[n1] += 1
        n1_count[n1] += 1

        overall_count[n2] += 1
        n2_count[n2] += 1

    for value in overall_count.values():
        assert np.isclose(value, 6000, 200)

    for bucket in [a_count, pos_count, n1_count, n2_count]:
        for value in bucket.values():
            assert np.isclose(value, 1700, atol=125)


def test_triplet_gen_construction():
    x, y = simple_test_set()
    qg = TripletGenerator(x,
                          y,
                          preprocessing=preprocessing,
                          augmentation=augmentation)


def test_triplet_get_batch():
    x, y = simple_test_set()
    qg = TripletGenerator(x,
                          y,
                          preprocessing=preprocessing,
                          augmentation=augmentation,
                          batch_size=10240)

    batch = qg.get_batch(0)

    anc = batch.values["anchor_example_f1"]
    neg = batch.values["neg_example_f1"]
    pos = batch.values["pos_example_f1"]

    assert len(anc) == 10240
    assert len(pos) == 10240
    assert len(neg) == 10240

    # In our simple test, all items for the same class have the same value,
    # +/- .0625
    np.allclose(pos, anc, atol=.063)

    # None of the other pairs of elements should be the same, since their
    # classes are different, and thus should have different values.
    overall_count = collections.defaultdict(int)
    a_count = collections.defaultdict(int)
    neg_count = collections.defaultdict(int)
    pos_count = collections.defaultdict(int)

    for a, n, p in zip(anc, neg, pos):
        assert n != p, "Bad tuple neg==pos: %s %s %s" % ((a, n, p))
        assert n != a, "Bad tuple neg==anchor: %s %s %s" % ((a, n, p))
        assert a == np.round(p), "Bad tuple anchor!=pos: %s %s %s" % (
            (a, n, p))

        overall_count[a] += 1
        a_count[a] += 1

        overall_count[p] += 1
        pos_count[p] += 1

        overall_count[n] += 1
        neg_count[n] += 1

    for value in overall_count.values():
        assert np.isclose(value, 6000, 200)

    for bucket in [a_count, pos_count, neg_count]:
        for value in bucket.values():
            assert np.isclose(value, 1700, atol=125)
