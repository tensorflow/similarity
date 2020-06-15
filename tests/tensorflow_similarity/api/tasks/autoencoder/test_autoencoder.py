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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask, ExampleDecoder
from tensorflow_similarity.utils.model_utils import *


def tower_model():
    i = Input(name="intinput", dtype=tf.float32, shape=(32, ))
    i2 = Input(name="intinputv", dtype=tf.float32, shape=(32, ))
    o = Dense(64)(i)
    o2 = Dense(64)(i2)

    o = Concatenate()([o, o2])

    o = Dense(64)(o)
    m = Model(inputs=[i, i2], outputs=o, name="tower_model")
    return m


def autoencoder_task(tower_model, **kwargs):
    task = AutoencoderTask("ae",
                           tower_model,
                           ExampleDecoder(name="decode"),
                           tower_names=["anchor"],
                           field_names=["intinput"],
                           **kwargs)
    task.build(compile=True)
    return task


def generate_autoencoder_input(N=10):

    x = {
        "anchor_intinput": np.random.rand(N, 32),
        "pos_intinput": np.random.rand(N, 32),
        "neg1_intinput": np.random.rand(N, 32),
        "neg2_intinput": np.random.rand(N, 32),
        "anchor_intinputv": np.random.rand(N, 32),
        "pos_intinputv": np.random.rand(N, 32),
        "neg1_intinputv": np.random.rand(N, 32),
        "neg2_intinputv": np.random.rand(N, 32)
    }

    x_aug = {}
    for k, v in x.items():
        x_aug[k] = v + 1.0

    y = {}

    batch = Batch()
    batch.add_raw_features("main", x)
    batch.add_features("main", x_aug)
    batch.add_labels("main", y)

    return batch


def test_autoencoder_build_fit_predict():
    batch = generate_autoencoder_input(1)
    model = tower_model()
    model.summary()
    task = autoencoder_task(model)

    print(model.inputs)

    task.update_batch(batch)
    inputs = batch.values
    labels = batch.labels

    print(inputs)
    print(labels)

    task.task_model.summary()
    task.task_model.fit(batch.values, batch.labels, epochs=3, verbose=0)
    task.task_model.predict(batch.values)


def test_autoencoder_generate_inputs_no_aug():
    batch = generate_autoencoder_input()
    task = autoencoder_task(tower_model(), input_feature_type="raw")
    task.update_batch(batch)
    inputs = batch.values
    labels = batch.labels

    assert np.allclose(inputs["ae_anchor_intinput"],
                       batch.raw_values["anchor_intinput"])
    assert np.allclose(inputs["ae_anchor_intinputv"],
                       batch.raw_values["anchor_intinputv"])


def test_autoencoder_generate_inputs_aug():
    batch = generate_autoencoder_input()
    task = autoencoder_task(tower_model(), input_feature_type="augmented")

    task.update_batch(batch)
    inputs = batch.values
    labels = batch.labels

    assert np.allclose(inputs["ae_anchor_intinput"],
                       batch.values["anchor_intinput"])
    assert np.allclose(inputs["ae_anchor_intinputv"],
                       batch.values["anchor_intinputv"])


def test_autoencoder_generate_labels_no_aug():
    batch = generate_autoencoder_input()
    model = tower_model()
    task = autoencoder_task(model, input_feature_type="raw")

    task.update_batch(batch)
    inputs = batch.values
    labels = batch.labels

    assert np.allclose(np.zeros_like(batch.values["anchor_intinput"]),
                       labels["ae_anchor_intinput_out"])


def test_autoencoder_generate_labels_aug():
    batch = generate_autoencoder_input()
    task = autoencoder_task(tower_model(), input_feature_type="augmented")

    task.update_batch(batch)
    inputs = batch.values
    labels = batch.labels

    assert np.allclose(np.zeros_like(batch.values["anchor_intinput"]),
                       labels["ae_anchor_intinput_out"])


if __name__ == '__main__':
    from tensorflow.python import debug as tf_debug
    from tensorflow.python import Session

    sess = Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    tf.keras.backend.set_session(sess)
    test_autoencoder_build_fit_predict()
