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
from tensorflow.keras.models import Model


class ParameterizedModel(abc.ABC, Model):
    def __init__(self, model=None, **kwargs):
        if not model:
            model = self.build_model(inputs, outputs, **kwargs)
        super(ParameterizedModel, self).__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            **kwargs)
        self.config = kwargs

    def _index(self, layer_list):
        shapes = {}
        dtypes = {}
        by_name = {}

        for layer in layer_list:
            name = layer.name
            shapes[name] = layer.shape
            dtypes[name] = layer.dtype
            by_name[name] = layer

        return shapes, dtypes, by_name

    @abc.abstractmethod
    def paramaterized_model(self,
                            inputs,
                            outputs):
        """Creates a Model instance representing the model for the given
        parameters.

        Arguments:
            inputs {[type]} -- [description]
            outputs {[type]} -- [description]

        Returns:
            tf.keras.models.Model -- The created model.
        """
        return
