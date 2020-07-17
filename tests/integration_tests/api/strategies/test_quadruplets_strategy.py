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

import tempfile
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import tempfile
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask, ExampleDecoder
from tensorflow_similarity.api.strategies.quadruplet_strategy import (
    HardQuadrupletLossStrategy, QuadrupletLossStrategy)
from tensorflow_similarity.api.tasks.utils_for_test import (
    gen_learnable_testdata, learnable_model)
from tensorflow.keras.models import load_model


def autoencoder(tower_model):
    return AutoencoderTask("test",
                           tower_model,
                           ExampleDecoder(),
                           tower_names=["anchor"],
                           field_names=["intinput"])


def _compute_cluster_distances(embeddings, y):
    inter = []
    intra = []

    for i in range(len(embeddings)):
        for j in range(i, len(embeddings)):
            l = embeddings[i]
            r = embeddings[j]
            d = np.linalg.norm(l - r, ord=2)
            if y[i] == y[j]:
                intra.append(d)
            else:
                inter.append(d)

    return np.average(intra), np.average(inter)


def test_learning():
    tmpdir = tempfile.mkdtemp()
    x, y = gen_learnable_testdata()
    tower_model = learnable_model()
    model = QuadrupletLossStrategy(tower_model,
                                   hard_mining=False,
                                   hard_mining_directory=tempfile.mkdtemp())

    embeddings = model.predict(x)

    (avg_intracluster_distance, _) = _compute_cluster_distances(embeddings, y)

    model.fit(x,
              y,
              prewarm_epochs=0,
              epochs=50,
              verbose=0,
              generator_workers=1)
    embeddings = model.predict(x)

    print(embeddings)

    (fit_avg_intracluster_distance,
     fit_avg_intercluster_distance) = _compute_cluster_distances(
         embeddings, y)

    assert fit_avg_intracluster_distance < fit_avg_intercluster_distance
    assert fit_avg_intercluster_distance - fit_avg_intracluster_distance > .1
    assert fit_avg_intracluster_distance < avg_intracluster_distance


def test_learning_w_serialization(tmpdir):

    model_path = tmpdir.join("serialized_model.h5")

    x, y = gen_learnable_testdata()
    tower_model = learnable_model()
    model = QuadrupletLossStrategy(tower_model,
                                   hard_mining=False,
                                   hard_mining_directory=tempfile.mkdtemp())

    embeddings = model.predict(x)

    (avg_intracluster_distance, _) = _compute_cluster_distances(embeddings, y)

    model.fit(x,
              y,
              prewarm_epochs=0,
              epochs=25,
              verbose=0,
              generator_workers=1)
    embeddings_pre_serialization = model.predict(x)
    model.save(model_path)
    model = load_model(model_path)
    embeddings_post_serialization = model.predict(x)

    assert np.allclose(embeddings_post_serialization,
                       embeddings_pre_serialization)

    model.fit(x,
              y,
              prewarm_epochs=0,
              epochs=25,
              verbose=0,
              generator_workers=1)

    embeddings = model.predict(x)

    (fit_avg_intracluster_distance,
     fit_avg_intercluster_distance) = _compute_cluster_distances(
         embeddings, y)

    assert fit_avg_intracluster_distance < fit_avg_intercluster_distance
    assert fit_avg_intercluster_distance - fit_avg_intracluster_distance > .1
    assert fit_avg_intracluster_distance < avg_intracluster_distance


def test_learning_hard_mining():
    tmpdir = tempfile.mkdtemp()
    x, y = gen_learnable_testdata()
    tower_model = learnable_model()
    model = HardQuadrupletLossStrategy(
        tower_model,
        hard_mining=True,
        hard_mining_directory=tempfile.mkdtemp())

    embeddings = model.predict(x)
    (avg_intracluster_distance, _) = _compute_cluster_distances(embeddings, y)

    assert avg_intracluster_distance > .01

    model.fit(x,
              y,
              prewarm_epochs=0,
              epochs=1000,
              callbacks=[
                  EarlyStopping(monitor="loss",
                                mode='min',
                                min_delta=0.00000001,
                                patience=100)
              ],
              verbose=0,
              generator_workers=1)

    embeddings = model.predict(x)
    (fit_avg_intracluster_distance,
     fit_avg_intercluster_distance) = _compute_cluster_distances(
         embeddings, y)

    assert fit_avg_intracluster_distance < fit_avg_intercluster_distance
    assert fit_avg_intercluster_distance - fit_avg_intracluster_distance > .1
    assert fit_avg_intracluster_distance < avg_intracluster_distance


if __name__ == '__main__':
    test_learning()
