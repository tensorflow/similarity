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
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.python import debug as tf_debug

from tensorflow_similarity.api.tasks.inference import InferenceTask


def sum_model():
    i = Input(shape=(1,), dtype=tf.float32, name="x")
    i2 = Input(shape=(1,), dtype=tf.float32, name="y")
    o = Add()([i, i2])
    m = Model(inputs=[i, i2], outputs=o)
    return m


def test_inference():
    model = sum_model()
    examples = {
        "x": np.array([[1], [2], [3]]),
        "y": np.array([[1], [2], [3]]),
    }

    infer = InferenceTask("inference", model)
    infer.build(compile=True)
    infer.task_model.summary()

    out = infer.predict(examples)

    expected = [[2.0], [4.0], [6.0]]

    assert np.allclose(expected, out)

    out_dict = infer.interpret_predictions(out)

    assert np.allclose(expected, out_dict["embedding"])
