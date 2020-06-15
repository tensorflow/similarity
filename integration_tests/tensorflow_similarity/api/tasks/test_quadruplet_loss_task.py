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
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.python import debug as tf_debug

from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.engine.task import MetaTask
from tensorflow_similarity.api.generators.task_based_generator import TaskBasedGenerator
from tensorflow_similarity.api.tasks.quadruplet_loss_task import QuadrupletLossTask
from tensorflow_similarity.api.tasks.utils_for_test import *


def test_learning():
    x, y = gen_learnable_testdata()

    quadloss_task = MetaTask(
        "quadruplet_loss", learnable_model(),
        QuadrupletLossTask(x, y, learnable_model(), hard_mining=False),
        optimizer=tf.keras.optimizers.Adam(lr=.001))

    quadloss_task.build(compile=True)

    quadloss_task.task_model.summary()
    model = quadloss_task.task_model

    model.summary()

    tbg = TaskBasedGenerator(quadloss_task, [])
    x, y = tbg[0]
    p = quadloss_task.interpret_predictions(model.predict(x))
    trips = p["triplets"]
    trips = np.array(trips).flatten().tolist()
    quads = p["quadruplets"]
    quads = np.array(quads).flatten().tolist()

    res = model.predict(x)

    res = quadloss_task.interpret_predictions(res)

    assert np.average(res["triplets"]) > 0.0
    assert np.average(res["quadruplets"]) > 0.0

    model.fit(
        x,
        y,
        epochs=1000,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss")],
        verbose=0
    )

    res = model.predict(x)
    res = quadloss_task.interpret_predictions(res)

    assert np.average(res["triplets"]) < 0.05
    assert np.average(res["quadruplets"]) < 0.05
