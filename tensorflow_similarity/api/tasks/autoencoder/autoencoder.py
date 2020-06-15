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

import abc

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model, model_from_config
from tensorflow.keras.utils import deserialize_keras_object as deserialize
from tensorflow.keras.utils import serialize_keras_object as serialize

from tensorflow_similarity.api.engine.parameterized_model import ParameterizedModel
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.task import AuxillaryTask, Task
from tensorflow_similarity.api.engine.decoder import Decoder, SimpleDecoder
from tensorflow_similarity.utils.model_utils import (
    clone_model_input,
    clone_model_inputs,
    compute_size,
    get_input_names,
    index_inputs,
    layer_shape)
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils.config_utils import register_custom_object


class ExampleDecoder(SimpleDecoder):
    def build_reconstruction_model(self):
        embedding = self.create_embedding_input()
        o = Dense(1024, activation="relu")(embedding)
        o = Dense(256, activation="relu")(o)
        m = Model(inputs=embedding, outputs=o, name="ExampleDecoder")
        m.compile(loss="mae", optimizer="adam")
        return m


register_custom_object("ExampleDecoder", ExampleDecoder)


class AutoencoderTask(AuxillaryTask):
    def __init__(self,
                 name,
                 tower_model,
                 decoder,
                 tower_names,
                 field_names,
                 loss="mae",
                 loss_weight=1.0,
                 input_preprocessing=None,
                 target_preprocessing=None,
                 input_feature_type="augmented",
                 target_feature_type="augmented"):
        super(AutoencoderTask, self).__init__(name, tower_model)

        self.name = name
        self.tower_names = tower_names
        self.field_names = field_names
        self.loss = loss
        self.tower_model = tower_model
        if not isinstance(tower_model, Model):
            self.tower_model = model_from_config(self.tower_model)
        self.loss_weight = loss_weight
        self.all_input_fields = get_input_names(self.tower_model)

        self.decoder = decoder
        if not isinstance(decoder, Decoder):
            self.decoder = deserialize(self.decoder)

        self.task_model = None

        self.task_inputs = []
        self.task_input_names = []

        self.task_outputs = []
        self.task_output_names = []
        self.task_losses = {}

        self.input_feature_type = input_feature_type
        self.input_preprocessing = input_preprocessing
        if not isinstance(self.input_preprocessing, Preprocessing):
            self.input_preprocessing = deserialize(self.input_preprocessing)

        self.target_feature_type = target_feature_type
        self.target_preprocessing = target_preprocessing
        if not isinstance(self.target_preprocessing, Preprocessing):
            self.target_preprocessing = deserialize(self.target_preprocessing)

        input_dictionary = index_inputs(self.tower_model)

        for field_name in field_names:
            assert field_name in input_dictionary, \
                "%s is not a field in the model. Known fields: %s" % (
                    field_name, input_dictionary)

    def _input_name(self, tower, field):
        return "%s_%s_%s" % (self.name, tower, field)

    def _source_input_name(self, tower, field):
        return "%s_%s" % (tower, field)

    def _output_name(self, tower, field):
        return "%s_%s_%s_out" % (self.name, tower, field)

    def build_task(self):
        for tower_name in self.tower_names:
            input_names, inputs = clone_model_inputs(self.tower_model,
                                                     prefix="%s_%s_" %
                                                     (self.name, tower_name))

            for name, inp in zip(input_names, inputs):
                self._add_input(name, inp)

            embedding = self.tower_model(self.task_inputs)

            for field_name in self.field_names:
                input_name = self._input_name(tower_name, field_name)
                output_name = self._output_name(tower_name, field_name)

                input_layer = self.task_input_dict[input_name]

                decoder_model = self.decoder.build(layer_shape(embedding),
                                                   embedding.dtype,
                                                   layer_shape(input_layer),
                                                   input_layer.dtype)

                reconstruction_difference = decoder_model(
                    [embedding, input_layer])

                output = Rename(name=output_name)(reconstruction_difference)

                self._add_output(output_name, output, self.loss,
                                 self.loss_weight)

        self.task_model = Model(inputs=self.task_inputs,
                                outputs=self.task_outputs,
                                name=self.name)

    def update_batch(self, batch):
        """For numeric autoencoders, no manipulation of the inputs is necessary.
        For a string, for instance, some manipulation of the input will may be
        necessary, as there is no practical way to generate a string in
        tensorflow, so an intermediate format(e.g. ordinals) may be necessary.

        Args:
            batch (Batch) -- The input batch, so far.

        Returns:
            Batch: the input Batch, including features/labels for this task.
        """
        inputs = {}

        batch_size = -1
        for tower_name in self.tower_names:
            for field_name in self.all_input_fields:
                source_name = self._source_input_name(tower_name, field_name)
                input_name = self._input_name(tower_name, field_name)
                value = batch.get(source_name,
                                  val_type=self.input_feature_type)
                if self.input_preprocessing:
                    value = self.input_preprocessing(value)
                inputs[input_name] = value
                batch_size = np.shape(value)[0]

        batch.add_features(self.name, inputs)

        labels = {}
        for tower_name in self.tower_names:
            for field_name in self.field_names:
                source_name = self._source_input_name(tower_name, field_name)
                value = batch.get(source_name,
                                  val_type=self.input_feature_type)

                output_name = self._output_name(tower_name, field_name)
                labels[output_name] = np.zeros(np.shape(value),
                                               dtype=np.float32)

        batch.add_labels(self.name, labels)

    def get_config(self):
        return {
            "name": self.name,
            "tower_model": serialize(self.tower_model),
            "decoder": serialize(self.decoder),
            "tower_names": self.tower_names,
            "field_names": self.field_names,
            "loss": self.loss,
            "loss_weight": self.loss_weight,
            "input_preprocessing": serialize(self.input_preprocessing),
            "target_preprocessing": serialize(self.target_preprocessing),
            "input_feature_type": self.input_feature_type,
            "target_feature_type": self.target_feature_type
        }


register_custom_object("AutoencoderTask", AutoencoderTask)
