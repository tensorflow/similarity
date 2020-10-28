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


class Split(Layer):
    def __init__(self, N=4, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.N = 4

    def call(self, input_tensor):
        return tf.split(
            input_tensor, num_or_size_splits=self.N, axis=1)

    def build(self, input_shape):
        super(Split, self).build(input_shape)


register_custom_object("Split", Split)
