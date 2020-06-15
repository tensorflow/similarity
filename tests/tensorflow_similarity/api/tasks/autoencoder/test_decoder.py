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
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow_similarity.api.engine.decoder import SimpleDecoder
from tensorflow_similarity.utils.model_utils import compute_size


class TestDecoder(SimpleDecoder):
    def build_reconstruction_model(self):
        embedding = self.create_embedding_input()
        o = Dense(64)(embedding)
        o = self.feature_shaped_dense(o)
        m = Model(embedding, o, name="reconstruction")
        m.summary()
        return m


def test_simple_decoder_recon():
    td = TestDecoder()
    td.set_parameters((2, ), tf.float32, (1, ), tf.float32)
    decoder_model = td.build_reconstruction_model()
    decoder_model.summary()
    decoder_model.compile(loss="mse", optimizer="adam")

    emb = np.zeros((1, 2))
    feature = np.array([[1.0]])
    label = np.array([1.0])

    decoder_model.fit(x=emb, y=label, epochs=1)


def test_simple_decoder():
    td = TestDecoder()
    decoder_model = td.build((2, ), tf.float32, (1, ), tf.float32)
    decoder_model.summary()
    decoder_model.compile(loss="mse", optimizer="adam")

    emb = np.zeros((1, 2))
    feature = np.array([[1.0]])
    label = np.array([1.0])

    decoder_model.fit(x=[emb, feature], y=label, epochs=1)


def test_simple_decoder():
    td = TestDecoder()
    decoder_model = td.build((2, ), tf.float32, (1, ), tf.float32)
    decoder_model.summary()
    decoder_model.compile(loss="mse", optimizer="adam")

    emb = np.zeros((1, 2))
    feature = np.array([[1.0]])
    label = np.array([1.0])

    decoder_model.fit(x=[emb, feature], y=label, epochs=1)
