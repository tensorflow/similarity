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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.tasks.image_cloze import (
    ExampleReconstructionModel, ImageClozeTask, blank_out_patch)
from tensorflow_similarity.api.tasks.utils_for_test import img_dataset, img_tower


def test_blank_out():
    raw = np.ones((4, 4, 3), dtype=np.float32)
    blanked = blank_out_patch(raw, (2, 2), (0, 0))

    for x in range(4):
        for y in range(4):
            for c in range(3):
                if x < 2 and y < 2:
                    assert np.isclose(blanked[x][y][c], 0.0)
                else:
                    assert np.isclose(blanked[x][y][c], 1.0)


def test_construction():
    batch = img_dataset()
    tower = img_tower()

    task = ImageClozeTask("cloze",
                          tower,
                          ExampleReconstructionModel(),
                          tower_names=["anchor"],
                          field_names=["image"])


def test_gen():
    batch = img_dataset()
    tower = img_tower()

    patch_size = (4, 4)
    max_damage_per_image = 4 * 4 * 3
    min_damage_per_image = 4 * 4 * 3 * .1

    task = ImageClozeTask("cloze",
                          tower,
                          ExampleReconstructionModel(),
                          patch_size=patch_size,
                          tower_names=["anchor"],
                          field_names=["image"])

    task.update_batch(batch)

    original = batch.values["anchor_image"]
    damaged = batch.values["cloze_anchor_image"]

    damage = original - damaged

    pixels_damaged = np.count_nonzero(damage)
    num_images = len(original)
    assert pixels_damaged == 4 * 4 * 3 * num_images
