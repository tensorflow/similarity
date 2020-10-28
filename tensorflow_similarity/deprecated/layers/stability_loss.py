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
import tensorflow.keras.backend as K
from tensorflow_similarity.distances import L2
from tensorflow_similarity.layers.utils import hinge
from tensorflow_similarity.utils.config_utils import register_custom_object


class PushLoss(Layer):
    def __init__(self,
                 margin=8.0,
                 distance=L2(),
                 **kwargs):
        super(PushLoss, self).__init__(**kwargs)
        self.distance = distance
        self.margin = margin

    def call(self, input_tensor):
        anchor, negative1, negative2, _ = tf.split(
            input_tensor, num_or_size_splits=4, axis=1)

        D_a_n1 = self.distance(anchor, negative1)
        D_a_n2 = self.distance(anchor, negative2)
        D_n1_n2 = self.distance(negative1, negative2)

        loss = hinge(3 * self.margin - (D_a_n1 + D_a_n2 + D_n1_n2))
        return tf.identity(loss, name="push")

    def build(self, input_shape):
        super(PushLoss, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("PushLoss", PushLoss)


class PullLoss(Layer):
    def __init__(self,
                 margin=1.0,
                 distance=L2(),
                 **kwargs):
        super(PullLoss, self).__init__(**kwargs)
        self.distance = distance
        self.margin = margin

    def call(self, input_tensor):
        anchor, _, _, positive = tf.split(
            input_tensor, num_or_size_splits=4, axis=1)

        D_a_p = self.distance(anchor, positive)

        loss = hinge(D_a_p - self.margin)
        return tf.identity(loss, name="pull")

    def build(self, input_shape):
        super(PullLoss, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("PullLoss", PullLoss)


class StabilityLoss(Layer):
    def __init__(self,
                 margin=1.0,
                 distance=L2(),
                 **kwargs):
        super(StabilityLoss, self).__init__(**kwargs)
        self.distance = distance
        self.margin = margin

    def call(self, input_tensor):
        anchor, negative1, negative2, _ = tf.split(
            input_tensor, num_or_size_splits=4, axis=1)

        D_a_n1 = self.distance(anchor, negative1)
        D_a_n2 = self.distance(anchor, negative2)
        D_n1_n2 = self.distance(negative1, negative2)

        loss = hinge(D_a_n1 / D_a_n2 - D_a_n2 / D_n1_n2 - self.margin)
        return tf.identity(loss, name="stability")

    def build(self, input_shape):
        super(StabilityLoss, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("StabilityLoss", StabilityLoss)
