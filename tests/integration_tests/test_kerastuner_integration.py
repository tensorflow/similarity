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

import copy
import json
import os
import pprint
import numpy as np
import tensorflow as tf
from absl import app, flags
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model, model_from_json

from kerastuner.abstractions import TENSORFLOW_UTILS as tf_utils
from kerastuner.distributions import Choice
from kerastuner.tuners import RandomSearch

from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow_similarity.api.strategies.quadruplet_strategy import *
from tensorflow_similarity.api.tasks.utils_for_test import (
    gen_learnable_testdata, learnable_model)


def hypertunable_model(embedding_size=4):
    hidden_layer_size = Choice("size", [8, 12], group="default")
    i = Input(name="intinput", dtype=tf.float32, shape=(2, ))
    i2 = Input(name="intinputv", dtype=tf.float32, shape=(2, ))
    o = Concatenate()([i, i2])
    o = Dense(hidden_layer_size, activation="relu")(o)
    o = Dense(embedding_size, activation="sigmoid")(o)
    m = Model(inputs=[i, i2], outputs=o, name="learnable_model")
    return m


def test_deserialize_with_kerastuner(tmpdir):
    flags.FLAGS(["pytest"])

    tmpdir = str(tmpdir)
    save_path = os.path.join(tmpdir, "model")

    out_dir = tmpdir

    model_path = "%s%s" % (save_path, "-config.json")
    weights_path = "%s%s" % (save_path, "-weights.h5")
    tmp_dir = os.path.join(tmpdir, "tmp")
    os.makedirs(tmp_dir)

    i = Input(shape=[1])
    o = Dense(1)(i)
    model = Model(i, o)

    sim = SimilarityModel(model,
                          augmentation=None,
                          preprocessing=None,
                          strategy="quadruplet_loss",
                          optimizer="adam")

    print("Saving to %s (tmp %s)" % (out_dir, tmp_dir))
    tf_utils.save_model(
        sim, save_path, export_type="keras", tmp_path=tmp_dir)

    ser_sim = None
    with open(model_path, "rt") as f:
        ser_sim = model_from_json(f.read())

    ser_sim.load_weights(weights_path)

    ser = serialize(sim)
    ser2 = serialize(ser_sim)

    serf = pprint.pformat(ser)
    ser2f = pprint.pformat(ser2)

    # Depending on how the model is loaded, you can get input shapes that are
    # lists or tuples, but otherwise function identically.
    serf = serf.replace("(", "[")
    serf = serf.replace(")", "]")
    ser2f = ser2f.replace("(", "[")
    ser2f = ser2f.replace(")", "]")

    assert serf == ser2f


def test_kerastuner_integration():
    def model_fn():
        return SimHash(hypertunable_model(),
                       augmentation=None,
                       preprocessing=None,
                       strategy="quadruplet_loss",
                       optimizer=Adam(lr=.001)).model

    flags.FLAGS(["pytest"])

    x, y = gen_learnable_testdata(copies=100)
    x_targets, y_targets = gen_learnable_testdata(copies=1)
    x_test, y_test = gen_learnable_testdata(copies=100)

    x_validation = {
        "targets": x_targets,
        "tests": x_test,
    }

    y_validation = {
        "targets": y_targets,
        "tests": y_test,
    }

    tuner = RandomSearch(model_fn,
                         objective='loss',
                         epoch_budget=6,
                         max_epochs=3)

    tuner.search(x, y, similarity_validation_data=(
        x_validation, y_validation), generator_workers=1)

    _, _, models = tuner.get_best_models(1)
    model = models[0]

    model.summary()
    tuner.save_best_model(export_type="keras_bundle")

    print(model.get_config())


if __name__ == '__main__':
    test_kerastuner_integration()
