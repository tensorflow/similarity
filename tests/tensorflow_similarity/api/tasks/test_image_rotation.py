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
from tensorflow_similarity.api.tasks.image_rotation import (ImageRotationTask,
                                                            rotate_img,
                                                            rotate_imgs)
from tensorflow_similarity.api.tasks.utils_for_test import (
    img_dataset, img_tower, make_small_img)


def test_rotate_img():
    a = make_small_img(1, 2, 3, 4)

    rotation_270 = make_small_img(3, 1, 4, 2)
    rotation_180 = make_small_img(4, 3, 2, 1)
    rotation_90 = make_small_img(2, 4, 1, 3)

    a_0 = rotate_img(a, 0)
    a_90 = rotate_img(a, 1)
    a_180 = rotate_img(a, 2)
    a_270 = rotate_img(a, 3)

    a_neg_0 = rotate_img(a, 0, reverse=True)
    a_neg_90 = rotate_img(a, 1, reverse=True)
    a_neg_180 = rotate_img(a, 2, reverse=True)
    a_neg_270 = rotate_img(a, 3, reverse=True)

    assert np.allclose(a, a_0)
    assert np.allclose(rotation_90, a_90)
    assert np.allclose(rotation_180, a_180)
    assert np.allclose(rotation_270, a_270)

    assert np.allclose(a, a_neg_0)
    assert np.allclose(rotation_90, a_neg_270)
    assert np.allclose(rotation_180, a_neg_180)
    assert np.allclose(rotation_270, a_neg_90)


def test_rotate_img_float():
    a = make_small_img(1 / 4.0, 2 / 4.0, 3 / 4.0, 4 / 4.0, dtype=np.float)
    rotation_270 = make_small_img(
        3 / 4.0,
        1 / 4.0,
        4 / 4.0,
        2 / 4.0,
        dtype=np.float)
    rotation_180 = make_small_img(
        4 / 4.0,
        3 / 4.0,
        2 / 4.0,
        1 / 4.0,
        dtype=np.float)
    rotation_90 = make_small_img(
        2 / 4.0,
        4 / 4.0,
        1 / 4.0,
        3 / 4.0,
        dtype=np.float)

    a_0 = rotate_img(a, 0)

    a_90 = rotate_img(a, 1)
    a_180 = rotate_img(a, 2)
    a_270 = rotate_img(a, 3)

    a_neg_0 = rotate_img(a, 0, reverse=True)
    a_neg_90 = rotate_img(a, 1, reverse=True)
    a_neg_180 = rotate_img(a, 2, reverse=True)
    a_neg_270 = rotate_img(a, 3, reverse=True)

    assert np.allclose(a, a_0, atol=.01)
    assert np.allclose(rotation_90, a_90, atol=.01)
    assert np.allclose(rotation_180, a_180, atol=.01)
    assert np.allclose(rotation_270, a_270, atol=.01)

    assert np.allclose(a, a_neg_0, atol=.01)
    assert np.allclose(rotation_90, a_neg_270, atol=.01)
    assert np.allclose(rotation_180, a_neg_180, atol=.01)
    assert np.allclose(rotation_270, a_neg_90, atol=.01)


def test_rotate_imgs():
    a = np.array([
        make_small_img(1, 2, 3, 4),
        make_small_img(2, 4, 6, 8),
        make_small_img(3, 6, 9, 12),
        make_small_img(4, 8, 12, 16)
    ])

    rotated = rotate_imgs(a, [0, 1, 2, 3])

    expected = [
        make_small_img(1, 2, 3, 4),
        make_small_img(4, 8, 2, 6),
        make_small_img(12, 9, 6, 3),
        make_small_img(12, 4, 16, 8)
    ]

    assert np.allclose(expected, rotated)


def test_reverse_rotate_imgs():
    a = np.array([
        make_small_img(1, 2, 3, 4),
        make_small_img(2, 4, 6, 8),
        make_small_img(3, 6, 9, 12),
        make_small_img(4, 8, 12, 16)
    ])

    rotated = rotate_imgs(a, [0, 1, 2, 3])
    reversed = rotate_imgs(rotated, [0, 1, 2, 3], reverse=True)

    d = reversed - a
    d = np.sum(np.abs(d))
    assert np.allclose(a, reversed, atol=.05)


def test_construction():
    tower = img_tower()

    _ = ImageRotationTask(
        "image_rotation",
        tower,
        tower_names=["anchor"],
        field_names=["image"])


def test_gen():
    batch = img_dataset()
    tower = img_tower()

    task = ImageRotationTask(
        "image_rotation",
        tower,
        tower_names=["anchor"],
        field_names=["image"])

    task.update_batch(batch)

    original = batch.values["anchor_image"]
    rotated = batch.values["image_rotation_anchor_image"]
    rotations_labels = batch.labels["image_rotation_anchor_image_out"]

    rotations_labels = np.argmax(rotations_labels, axis=1)
    reconstructed = rotate_imgs(rotated, rotations_labels, reverse=True)

    d = np.square(original - reconstructed)
    d = np.sum(d)
    np.allclose(original.shape, reconstructed.shape)
    assert d < .001
