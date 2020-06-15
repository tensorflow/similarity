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

    x, y = gen_learnable_testdata(copies=100)
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    moirai = SimHash(learnable_model(),
                     augmentation=None,
                     preprocessing=None,
                     strategy="quadruplet_loss",
                     optimizer=Adam(lr=.001))

    results = moirai.fit(x,
                         y,
                         prewarm_epochs=0,
                         epochs=5,
                         verbose=0,
                         generator_workers=1)

    _ = moirai.predict(x)

    metrics = moirai.evaluate(x_test, y_test, x_targets, y_targets)

    print(results.history["loss"])

    assert results.history["loss"][-1] < results.history["loss"][0]
    assert metrics['validation_set_max_rank'] <= 5.0


def test_hard_quad_loss_via_api():
    flags.FLAGS(["pytest"])

    x, y = gen_learnable_testdata(copies=100)
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    moirai = SimHash(learnable_model(),
                     augmentation=None,
                     preprocessing=None,
                     strategy="hard_quadruplet_loss",
                     optimizer=Adam(lr=.001))

    results = moirai.fit(x,
                         y,
                         prewarm_epochs=0,
                         epochs=5,
                         verbose=0,
                         generator_workers=1)

    metrics = moirai.evaluate(x_test, y_test, x_targets, y_targets)

    assert np.isclose(metrics['validation_set_max_rank'], 0.0, atol=4)
    assert np.isclose(metrics['validation_set_top1_acc'], 100.0, atol=50)


def test_stable_hard_quad_loss_via_api():
    flags.FLAGS(["pytest"])

    x, y = gen_learnable_testdata(copies=100)
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    moirai = SimHash(learnable_model(embedding_size=64),
                     augmentation=None,
                     preprocessing=None,
                     strategy="stable_hard_quadruplet_loss",
                     optimizer=Adam(lr=.0001))

    moirai.fit(x,
               y,
               prewarm_epochs=0,
               epochs=5,
               verbose=1,
               generator_workers=1)

    metrics = moirai.evaluate(x_test, y_test, x_targets, y_targets)

    assert np.isclose(metrics['validation_set_max_rank'], 0.0, atol=4)
    assert np.isclose(metrics['validation_set_top3_acc'], 100.0, atol=1)


if __name__ == '__main__':
    test_quad_loss_via_api()
