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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from tensorflow_similarity.distances import L2
from tensorflow_similarity.layers.utils import hinge
from tensorflow_similarity.utils.config_utils import register_custom_object


class TripletLoss(Layer):
    def __init__(self,
                 margin=.2,
                 distance=L2(),
                 **kwargs):
        super(TripletLoss, self).__init__(**kwargs)

        self.distance = distance
        self.margin = margin

    def call(self, input_tensor):
        anchor, negative1, positive = tf.split(
            input_tensor, num_or_size_splits=3, axis=1)

        D_pa = self.distance(positive, anchor)
        D_an1 = self.distance(anchor, negative1)

        return hinge(K.constant(self.margin, dtype='float32') + D_pa - D_an1)

    def build(self, input_shape):
        super(TripletLoss, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("TripletLoss", TripletLoss)


class TripletLossAddon(Layer):
    def __init__(self,
                 margin=.2,
                 distance=L2(),
                 **kwargs):
        super(TripletLossAddon, self).__init__(**kwargs)
        self.distance = distance
        self.margin = margin

    def call(self, input_tensor):
        anchor, negative1, _, positive = tf.split(
            input_tensor, num_or_size_splits=4, axis=1)

        D_pa = self.distance(positive, anchor)
        D_an1 = self.distance(anchor, negative1)

        return hinge(D_pa - D_an1 + self.margin)

    def build(self, input_shape):
        super(TripletLossAddon, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("TripletLossAddon", TripletLossAddon)


class QuadrupletLossAddon(Layer):
    def __init__(self,
                 triplet_loss_margin=.2,
                 quadruplet_loss_margin=.1,
                 distance=L2(),
                 **kwargs):
        super(QuadrupletLossAddon, self).__init__(**kwargs)

        self.distance = distance
        self.triplet_loss_margin = triplet_loss_margin
        self.quadruplet_loss_margin = quadruplet_loss_margin

    def call(self, input_tensor):
        anchor, negative1, negative2, positive = tf.split(
            input_tensor, num_or_size_splits=4, axis=1)

        D_pa = self.distance(positive, anchor)
        D_n1n2 = self.distance(negative1, negative2)

        quad_loss_addon = hinge(
            D_pa - D_n1n2 + self.quadruplet_loss_margin)
        return tf.identity(quad_loss_addon, name="quadruplet_loss")

    def build(self, input_shape):
        super(QuadrupletLossAddon, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (1,)


register_custom_object("QuadrupletLossAddon", QuadrupletLossAddon)
