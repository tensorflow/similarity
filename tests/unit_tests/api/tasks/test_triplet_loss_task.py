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


def test_triplet_loss_build():
    x, y = gen_testdata()

    tripletloss_task = TripletLossTask(x, y, tower_model(), hard_mining=False)
    tripletloss_task.build()


def test_triplet_loss_gen():
    x, y = gen_testdata()

    tripletloss_task = TripletLossTask(x, y, tower_model(), hard_mining=False)
    tripletloss_task.build()

    batch = tripletloss_task.get_main_batch(0)

    labels = ["triplets", "hard_tuple_fraction",
              "meta_is_hard", "anchor_idx_out", "pos_idx_out", "neg_idx_out"]

    for label in labels:
        assert np.array_equal(batch.labels[label].shape, (128, ))

    for tower in ["anchor", "pos", "neg"]:
        for feature in ["intinput", "intinputv"]:
            assert np.array_equal(
                batch.values["%s_%s" % (tower, feature)].shape, (128, 2))
            assert np.array_equal(
                batch.raw_values["%s_%s" % (tower, feature)].shape, (128, 2))


def test_triplet_loss_fit():
    x, y = gen_testdata()

    tripletloss_task = TripletLossTask(x, y, tower_model(), hard_mining=False)
    tripletloss_task.build(compile=True)

    tripletloss_task.task_model.summary()
    model = tripletloss_task.task_model

    model.summary()

    tbg = TaskBasedGenerator(tripletloss_task, [])
    x, y = tbg[0]

    model.fit(x, y, epochs=1, verbose=0)


def test_meta_triplet_loss_fit():
    x, y = gen_testdata()

    tripletloss_task = MetaTask(
        "triplet_loss", tower_model(),
        TripletLossTask(x, y, tower_model(), hard_mining=False))

    tripletloss_task.build(compile=True)

    tripletloss_task.task_model.summary()
    model = tripletloss_task.task_model

    model.summary()

    tbg = TaskBasedGenerator(tripletloss_task, [])
    x, y = tbg[0]
    model.fit(x, y, epochs=1, verbose=0)
