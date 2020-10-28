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

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow_similarity.utils.config_utils import register_custom_object


class Rename(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Rename, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rename, self).build(input_shape)

    def call(self, x):
        return tf.identity(x, name=self.name)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {}


register_custom_object("Rename", Rename)
