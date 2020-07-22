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
from tensorflow.python import Session
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow_similarity.api.engine.decoder import SimpleDecoder
from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.tasks.image_cloze import (
    ExampleReconstructionModel, ImageClozeTask, blank_out_patch)
from tensorflow_similarity.api.tasks.utils_for_test import img_dataset
from tensorflow.python import debug as tf_debug
from tensorflow.keras.utils import serialize_keras_object as serialize

tf.compat.v1.disable_eager_execution()


def img_tower(dim=5):
    i = Input(shape=(dim, dim, 3), dtype=np.float32, name="image")
    o = i
    o = SeparableConv2D(64, (3, 3), activation="relu")(o)
    o = SeparableConv2D(128, (3, 3), activation="relu")(o)
    o = Flatten()(o)
    o = Dense(128)(o)
    return Model(inputs=[i], outputs=[o], name="img_tower")


def test_fit():
    print("Yay")

    s = Session()
    #s = tf_debug.LocalCLIDebugWrapperSession(s)
    tf.keras.backend.set_session(s)
    # Adam_1/gradients/loss_1/cloze_anchor_image_out_loss/Mean_grad/DynamicStitch
    batch = img_dataset(5)
    tower = img_tower(5)

    patch_size = (4, 4)

    task = ImageClozeTask("cloze",
                          tower,
                          ExampleReconstructionModel("anchor_image"),
                          patch_size=patch_size,
                          tower_names=["anchor"],
                          field_names=["image"])
    task.build(compile=True)
    model = task.task_model
    model.summary()
    task.update_batch(batch)

    x_orig = batch.values["anchor_image"]
    x = batch.values["cloze_anchor_image"]
    y = batch.labels["cloze_anchor_image_out"]

    history = model.fit(x, y, epochs=100)

    assert history.history["loss"][-1] < .6


if __name__ == '__main__':
    test_fit()
