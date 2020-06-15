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
from PIL import Image
from tensorflow.keras.layers import Dense, Flatten, Input, SeparableConv2D
from tensorflow.keras.models import Model

from tensorflow_similarity.api.engine.generator import Batch

from tensorflow_similarity.api.tasks.image_cloze import blank_out_patch
from tensorflow_similarity.api.tasks.image_rotation import (
    ImageRotationTask,
    image_rotation_model,
    rotate_img,
    rotate_imgs)
from tensorflow_similarity.api.tasks.utils_for_test import img_dataset, img_tower, make_small_img


def test_fit():
    batch = img_dataset()
    tower = img_tower()

    task = ImageRotationTask(
        "image_rotation",
        tower,
        image_rotation_model,
        tower_names=["anchor"],
        field_names=["image"])
    task.build(compile=True)
    model = task.task_model
    task.update_batch(batch)

    x_orig = batch.values["anchor_image"]
    x = batch.values["image_rotation_anchor_image"]
    y = batch.labels["image_rotation_anchor_image_out"]

    history = model.fit(
        x,
        y,
        epochs=1000,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
        ],
        verbose=0)

    assert history.history["loss"][-1] < .5
