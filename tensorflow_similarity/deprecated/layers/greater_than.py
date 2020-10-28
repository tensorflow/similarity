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


class GreaterThan(Layer):
    """ Returns a tensor in the same shape as the input, where each value is
    1.0 if the element of the original tensor is greater than the threshold, or
    0.0 otherwise."""

    def __init__(self,
                 threshold=0.0,
                 **kwargs):
        super(GreaterThan, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, input_tensor):
        return tf.cast(
            tf.math.greater(input_tensor, tf.constant(self.threshold)),
            dtype=tf.float32)

    def build(self, input_shape):
        super(GreaterThan, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("GreaterThan", GreaterThan)
