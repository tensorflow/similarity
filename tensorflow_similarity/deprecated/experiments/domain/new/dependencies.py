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

import tensorflow_tensortext
import tensorflow_similarity
from tensorflow_similarity.layers import *
from tensorflow_similarity.experiments.domain.new.augment import DomainAugment
from tensorflow_similarity.utils.config_utils import register_custom_object
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer


class WrapArray(object):
    def __call__(self, x):
        return np.array([x])

    def get_config(self):
        return {"class_name": self.__class__.__name__, "config": {}}


register_custom_object("WrapArray", WrapArray)


class FakeHash(Layer):
    def __init__(self, **kwargs):
        super(FakeHash, self).__init__(**kwargs)
        self._name = "FakeHash"
        pass

    def build(self, input_shape):
        super(FakeHash, self).build(input_shape)

    def call(self, x):
        primes = [7307,
                  7309,
                  7321,
                  7331,
                  7333,
                  7349,
                  7351,
                  7369]

        stack = []
        for prime in primes:
            z = tf.strings.to_hash_bucket_fast(x, prime)
            z = tf.math.mod(z, 4)
            z = tf.one_hot(z, depth=4)
            stack.append(z)

        o = tf.concat(stack, axis=-1)
        o = tf.squeeze(o, axis=-2, name="embedding")
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 32)


register_custom_object("FakeHash", FakeHash)
