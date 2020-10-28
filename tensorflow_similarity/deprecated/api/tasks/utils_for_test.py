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

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow_similarity.utils.model_utils import compute_size
from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.engine.decoder import Decoder, SimpleDecoder
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask
import numpy as np

# TODO - many of these should be converted to fixtures.


def make_small_img(a, b, c, d, dtype=np.uint8):
    return np.array([[[a, a, a], [b, b, b]], [[c, c, c], [d, d, d]]],
                    dtype=dtype)


def img_tower():
    i = Input(shape=(32, 32, 3), dtype=np.float32, name="image")
    o = SeparableConv2D(64, (3, 3), activation="relu")(i)
    o = SeparableConv2D(128, (3, 3), activation="relu")(i)
    o = Flatten()(o)
    o = Dense(128)(o)
    return Model(inputs=[i], outputs=[o], name="img_tower")


def img_dataset(dim=32):
    imgs = []
    labels = []

    for r in range(1, 255, 62):
        for g in range(1, 255, 62):
            for b in range(1, 255, 62):
                label = "%d_%d_%d" % (r, g, b)
                rs = np.ones((dim, dim), dtype=np.float) * r / 255.0
                gs = np.ones((dim, dim), dtype=np.float) * g / 255.0
                bs = np.ones((dim, dim), dtype=np.float) * b / 255.0
                img = np.stack([rs, gs, bs], axis=-1)
                labels.append(label)
                imgs.append(img)

    batch = Batch()
    batch.add_features("cloze", {"anchor_image": imgs})
    batch.add_labels("cloze", {"anchor_targets": labels})
    return batch


def autoencoder_task(tower_model, **kwargs):
    task = AutoencoderTask("ae",
                           tower_model,
                           numeric_decoder_model,
                           tower_names=["anchor"],
                           field_names=["intinput"],
                           **kwargs)
    task.build(compile=True)
    return task


def tower_model():
    i = Input(name="intinput", dtype=tf.float32, shape=(2, ))
    i2 = Input(name="intinputv", dtype=tf.float32, shape=(2, ))
    o = Dense(64)(i)
    o2 = Dense(64)(i2)

    o = Concatenate()([o, o2])

    o = Dense(64)(o)
    m = Model(inputs=[i, i2], outputs=o, name="tower_model")
    return m


def test_model():
    i = Input(name="intinput", dtype=tf.float32, shape=(32, ))
    i2 = Input(name="intinputv", dtype=tf.float32, shape=(64, ))
    o = Dense(64)(i)
    o2 = Dense(64)(i2)

    o = Concatenate()([o, o2])

    o = Dense(64)(o)
    m = Model([i, i2], o, name="test_model")
    return m


def learnable_model(embedding_size=4):
    i = Input(name="intinput", dtype=tf.float32, shape=(2, ))
    i2 = Input(name="intinputv", dtype=tf.float32, shape=(2, ))
    o = Concatenate()([i, i2])
    o = Dense(8, activation="relu")(o)
    o = Dense(embedding_size, activation="sigmoid")(o)
    m = Model(inputs=[i, i2], outputs=o, name="learnable_model")
    return m


def gen_testdata():
    intinput = np.random.rand(10, 2)
    intinputv = np.random.rand(10, 2) + 2.0
    x = {"intinput": intinput, "intinputv": intinputv}
    y = np.random.randint(0, 10, size=10)

    return x, y


def gen_learnable_testdata(classes=6, copies=3):
    intinput = np.random.rand(classes * copies, 2)
    intinputv = np.random.rand(classes * copies, 2)

    y = []
    for i in range(classes):
        for _ in range(copies):
            y.append(i)

    for idx, y_ in enumerate(y):
        intinput[idx][0] = y_
        intinputv[idx][0] = y_

    intinput = np.array(intinput)
    intinputv = np.array(intinputv)

    x = {"intinput": intinput, "intinputv": intinputv}
    return x, y
