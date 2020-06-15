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

from tensorflow_similarity.architectures.tcn.base import TCN
from tensorflow_similarity.architectures import util
from tensorflow.keras import backend as K
from tensorflow_similarity.architectures.model_registry import register_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def tcn_v0(input_shape):
    i = Input(shape=input_shape['example'], name='example')
    o = TCN(return_sequences=False, nb_filters=256, nb_stacks=4)(i)
    o = Dense(
        256,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(o)
    m = Model(inputs=[i], outputs=[o])
    return m


register_model(tcn_v0)


def tcn_v1(input_shape):
    i = Input(shape=input_shape['example'], name='example')
    o = TCN(return_sequences=False, nb_filters=512, nb_stacks=5)(i)
    o = Dense(
        256,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(o)
    m = Model(inputs=[i], outputs=[o])
    return m


register_model(tcn_v1)

if __name__ == '__main__':
    tcn_v0({'examples': (32, 64)})
