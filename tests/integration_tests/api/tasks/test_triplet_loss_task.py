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
from tensorflow_similarity.api.tasks.triplet_loss_task import TripletLossTask
from tensorflow_similarity.api.tasks.utils_for_test import *


def test_learning():
    x, y = gen_learnable_testdata()

    triplet_loss_task = MetaTask(
        "triplet_loss", learnable_model(),
        TripletLossTask(x, y, learnable_model(), hard_mining=False),
        optimizer=tf.keras.optimizers.Adam(lr=.001))

    triplet_loss_task.build(compile=True)

    triplet_loss_task.task_model.summary()
    model = triplet_loss_task.task_model

    model.summary()

    tbg = TaskBasedGenerator(triplet_loss_task, [])
    x, y = tbg[0]
    p = triplet_loss_task.interpret_predictions(model.predict(x))
    trips = p["triplets"]
    trips = np.array(trips).flatten().tolist()
    res = model.predict(x)

    res = triplet_loss_task.interpret_predictions(res)

    assert np.average(res["triplets"]) > 0.0

    model.fit(
        x,
        y,
        epochs=1000,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss")],
        verbose=0
    )

    res = model.predict(x)
    res = triplet_loss_task.interpret_predictions(res)

    assert np.average(res["triplets"]) < 0.05
