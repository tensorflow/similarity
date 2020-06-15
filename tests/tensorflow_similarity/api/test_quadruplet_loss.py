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

from absl import app, flags
"""
Smoke tests - doesn't check the results, just check that everything compiles
and runs. For correctness tests, see integration_tests/moirai/api/test*
"""

import tensorflow as tf
from tensorflow_similarity.api.strategies.quadruplet_strategy import *
from tensorflow_similarity.api.engine.simhash import SimHash
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow
from tensorflow_similarity.api.tasks.utils_for_test import gen_learnable_testdata, learnable_model
import copy


def test_quad_loss_via_api():

    flags.FLAGS(["pytest"])

    x, y = gen_learnable_testdata()
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    moirai = SimHash(
        learnable_model(),
        augmentation=None,
        preprocessing=None,
        strategy="quadruplet_loss",
        optimizer=Adam(lr=.001))

    moirai.fit(
        x,
        y,
        prewarm_epochs=0,
        epochs=1,
        callbacks=[
            tensorflow.keras.callbacks.EarlyStopping(
                monitor="loss", mode='min', min_delta=0.00000001, patience=100)
        ],
        verbose=0
    )

    _ = moirai.predict(x)
    _ = moirai.evaluate(x_test, y_test, x_targets, y_targets)


def test_hard_quad_loss_via_api():
    flags.FLAGS(["pytest"])

    x, y = gen_learnable_testdata()
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    moirai = SimHash(
        learnable_model(),
        augmentation=None,
        preprocessing=None,
        strategy="hard_quadruplet_loss",
        optimizer=Adam(lr=.001))

    moirai.fit(
        x,
        y,
        prewarm_epochs=0,
        epochs=1,
        callbacks=[
            tensorflow.keras.callbacks.EarlyStopping(
                monitor="loss", mode='min', min_delta=0.00000001, patience=100)
        ],
        verbose=0)

    _ = moirai.predict(x)
    _ = moirai.evaluate(x_test, y_test, x_targets, y_targets)


def test_stable_hard_quad_loss_via_api():
    flags.FLAGS(["pytest"])

    x, y = gen_learnable_testdata()
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    moirai = SimHash(
        learnable_model(),
        augmentation=None,
        preprocessing=None,
        strategy="stable_hard_quadruplet_loss",
        optimizer=Adam(lr=.001))

    moirai.fit(
        x,
        y,
        prewarm_epochs=0,
        epochs=1,
        callbacks=[
            tensorflow.keras.callbacks.EarlyStopping(
                monitor="loss", mode='min', min_delta=0.00000001, patience=100)
        ],
        verbose=0)

    _ = moirai.predict(x)
    _ = moirai.evaluate(x_test, y_test, x_targets, y_targets)
