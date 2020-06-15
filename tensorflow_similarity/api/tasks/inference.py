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

import time

import six
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib

from tensorflow_similarity.api.engine.task import Task
from tensorflow_similarity.api.generators.tuples import (
    HardQuadrupletGenerator, QuadrupletGenerator)
from tensorflow_similarity.api.losses.no_loss import NoLoss
from tensorflow_similarity.utils.model_utils import *
from tensorflow_similarity.layers.greater_than import GreaterThan
from tensorflow_similarity.layers.loss_layers import QuadrupletLossAddon, TripletLossAddon
from tensorflow_similarity.layers.rename import Rename


class InferenceTask(Task):
    def __init__(self, *args, preprocessing=None, **kwargs):
        self.preprocessing = preprocessing
        super(InferenceTask, self).__init__(*args, **kwargs)

    def build_task(self):
        input_names, inputs = clone_model_inputs(self.tower_model, prefix="")

        for input_name, layer in zip(input_names, inputs):
            self._add_input(input_name, layer)

        devices = get_devices()
        with tf.device(devices[0]):
            output = self.tower_model(inputs)
            output = Rename(name="embedding")(output)

        self._add_output("embedding", output, "mae")
        self.task_model = Model(self.task_inputs, self.task_outputs)

    def predict(self, x):
        # Change the input from a list of raw arrays to a dictionary of
        # input -> array, if necessary
        if not isinstance(x, dict):
            x_ = {}
            for name, inp in zip(self.task_input_names, x):
                x_[name] = inp
            x = x_

        if not self.preprocessing:
            return self.task_model.predict(x)

        # Run the preprocessing on the now-guaranteed-to-be-a-dict input.
        examples = training_data_to_example_list(x)

        pp_examples = []

        # Run the preprocessing over each example.
        for example in examples:
            pp_examples.append(self.preprocessing.preprocess(example))

        # Convert it back to a dictionary of feature arrays.
        x = example_list_to_training_data(pp_examples)

        return self.task_model.predict(x)
