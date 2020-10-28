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


class ParameterizedModel(abc.ABC):
    def __init__(self):
        self.input_names = []
        self.input_shapes = []
        self.input_dtypes = []
        self.output_names = []
        self.output_shapes = []
        self.output_dtypes = []

        self.inputs_by_name = {}

        self.input_name_to_type = {}
        self.input_name_to_shape = {}
        self.output_name_to_type = {}
        self.output_name_to_shape = {}

    def set_parameters(
            self,
            input_names=[],
            input_shapes=[],
            input_dtypes=[],
            output_names=[],
            output_shapes=[],
            output_dtypes=[]):

        self.input_names = input_names
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.output_names = output_names
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes

        self.inputs_by_name = {}

        self.input_name_to_type = {}
        self.input_name_to_shape = {}
        self.output_name_to_type = {}
        self.output_name_to_shape = {}

        self._index()
        self._create_inputs()
        self.validate()

    def validate(self):
        pass

    def get_input(self, name):
        return self.inputs_by_name[name]

    def _create_inputs(self):
        for name, type, shape in zip(
                input_names, input_dtypes, input_shapes):
            i = Input(name=name, shape=shape, dtype=type)
            self.inputs_by_name[name] = i

    def _index(self):
        for name, type, shape in zip(
                input_names, input_dtypes, input_shapes):
            self.input_name_to_type[name] = type
            self.input_name_to_shape[name] = shape

        for name, type, shape in zip(
                output_names, output_dtypes, output_shapes):
            self.output_name_to_type[name] = type
            self.output_name_to_shape[name] = shape

    @abc.abstractmethod
    def build(self):
        """Builds a tf.keras.models.Model based on the input/output parameters
        specified, and returns it."""
        return
