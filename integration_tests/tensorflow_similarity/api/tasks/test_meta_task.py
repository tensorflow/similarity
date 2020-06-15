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
from tensorflow.keras.layers import Concatenate, Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.python import debug as tf_debug

from tensorflow_similarity.api.engine.generator import Batch
from tensorflow_similarity.api.engine.task import MetaTask
from tensorflow_similarity.api.generators.task_based_generator import TaskBasedGenerator
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask, ExampleDecoder

from tensorflow_similarity.utils.model_utils import *
from tensorflow_similarity.api.tasks.quadruplet_loss_task import QuadrupletLossTask
from tensorflow_similarity.api.tasks.utils_for_test import (
    gen_learnable_testdata, gen_testdata, learnable_model)


def test_meta_task_fit():
    x, y = gen_learnable_testdata()

    model = learnable_model()

    main_task = QuadrupletLossTask(x, y, model)
    ae_task = AutoencoderTask("autoencoder", model, ExampleDecoder(),
                              ["anchor"], ["intinput"])

    meta_task = MetaTask("quadloss_plus_autoencoder",
                         model,
                         main_task,
                         auxillary_tasks=[ae_task])
    meta_task.build(compile=True)

    batch = meta_task.generator.get_batch(0)

    res = meta_task.task_model.fit(batch.values, batch.labels, epochs=2)

    assert res.history["loss"][-1] < res.history["loss"][0]


if __name__ == '__main__':
    test_meta_task_fit()
