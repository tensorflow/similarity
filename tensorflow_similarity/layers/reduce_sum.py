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


class ReduceSum(Layer):
    """Layer wrapper for tf.reduce_sum."""

    def __init__(self, axis=None, **kwargs):
        """Construct a wrapper around tf.reduce_sum with the specified arguments.

        Args:
            axis (int or list of int, optional): Axis/axes in which to sum, with
                0 referring to the 0th non-batch axis. Defaults to None.
        """
        super(ReduceSum, self).__init__(**kwargs)

        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = axis + 1
        else:
            self.axis = [ax + 1 for ax in axis]

    def call(self, input_tensor):
        # Special case - a reduce_sum with no axis is a global reduce - in this
        # case we translate it sum all of the non-batch axes.
        if not self.axis:
            axes = [i for i in range(1, input_tensor.shape.rank)]
            return tf.math.reduce_sum(input_tensor, axis=axes)
        else:
            return tf.math.reduce_sum(input_tensor, axis=self.axis)


register_custom_object("ReduceSum", ReduceSum)
