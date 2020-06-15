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

import pytest
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow_similarity.layers.reduce_mean import ReduceMean
from tensorflow_similarity.utils.model_utils import compute_size
import numpy as np

BATCH_SIZE = 4


def model(input_shape, axes):
    i = Input(shape=input_shape, dtype=tf.float32)
    d = ReduceMean(axis=axes)(i)
    m = Model(i, d)
    m.summary()
    m.compile(loss="mse", optimizer="adam")
    return m


def test_reduce_mean_scalars():
    input_shape = [2]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones(BATCH_SIZE)

    m = model(input_shape, 0)
    m.fit(x, y)

    y_ = m.predict(x)
    assert np.array_equal(y_.shape, [BATCH_SIZE])
    assert np.array_equal(y_, [1] * BATCH_SIZE)


def test_reduce_mean_scalars_oob():
    input_shape = [3]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones(BATCH_SIZE)

    with pytest.raises(ValueError):
        m = model(input_shape, 1)


def test_reduce_mean_vectors():
    input_shape = [3, 5]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones([BATCH_SIZE, 5])

    m = model(input_shape, 0)
    m.summary()
    m.fit(x, y)

    y_ = m.predict(x)

    assert np.array_equal([BATCH_SIZE, 5], y_.shape)
    ev = np.ones((BATCH_SIZE, 5))
    assert np.array_equal(ev, y_)


def test_reduce_mean_vectors_to_scalars():
    input_shape = [3, 5]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones([BATCH_SIZE])

    m = model(input_shape, [0, 1])
    m.summary()
    m.fit(x, y)

    y_ = m.predict(x)

    assert np.array_equal([BATCH_SIZE], y_.shape)
    ev = np.ones((BATCH_SIZE))
    assert np.array_equal(ev, y_)


def test_reduce_mean_vectors_to_scalars_implicit():
    input_shape = [3, 5]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones([BATCH_SIZE])

    m = model(input_shape, None)
    m.summary()
    m.fit(x, y)

    y_ = m.predict(x)

    assert np.array_equal([BATCH_SIZE], y_.shape)
    ev = np.ones((BATCH_SIZE))
    assert np.array_equal(ev, y_)


def test_reduce_mean_matrices():
    input_shape = [3, 5, 6]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones([BATCH_SIZE, 5, 6])

    m = model(input_shape, 0)
    m.summary()
    m.fit(x, y)

    y_ = m.predict(x)

    assert np.array_equal([BATCH_SIZE, 5, 6], y_.shape)
    ev = np.ones((BATCH_SIZE, 5, 6))
    assert np.array_equal(ev, y_)


def test_reduce_mean_matrics_to_scalars():
    input_shape = [3, 5, 6]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones([BATCH_SIZE])

    m = model(input_shape, [0, 1, 2])
    m.summary()
    m.fit(x, y)

    y_ = m.predict(x)

    assert np.array_equal([BATCH_SIZE], y_.shape)
    ev = np.ones((BATCH_SIZE))
    assert np.array_equal(ev, y_)


def test_reduce_mean_matrics_to_scalars_implicit():

    input_shape = [3, 5, 6]
    x = np.ones([BATCH_SIZE] + input_shape)
    y = np.ones([BATCH_SIZE])

    m = model(input_shape, None)
    m.summary()
    m.fit(x, y)

    y_ = m.predict(x)

    assert np.array_equal([BATCH_SIZE], y_.shape)
    ev = np.ones((BATCH_SIZE))
    assert np.array_equal(ev, y_)
