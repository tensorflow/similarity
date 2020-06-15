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
from tensorflow_similarity.utils.config_utils import register_custom_object


class NoLoss(object):
    """
    NoLoss is a placeholder loss, which should always return 0.

    It is used to simplify the input and target handling process, as in
    some cases, not specifying losses for all output when using a dictionary
    of target values causes issues with Keras.

    TODO - check whether this is still necessary on the latest releases.
    """

    def __init__(self):
        self.__name__ = "NoLoss"

    def __call__(self, y_pred, y_actual):
        """Compute a 0 loss as a workaround to connect the Keras graph.

        Arguments:
            y_pred (tf.Tensor): The predicted label.
            y_actual (tf.Tensor): The actual label.

        Returns:
            tf.Tensor: A tensor that should always be 0.
        """
        if y_pred.dtype == tf.string:
            y_pred = tf.strings.length(y_pred)
            y_actual = tf.strings.length(y_actual)

        # TODO(b/141714580): current workaround for b/141714580.
        y_pred = tf.cast(y_pred, tf.int64)
        y_actual = tf.cast(y_actual, tf.int64)

        # Return 0, but in a fashion that keeps the graph connected.
        return tf.cast(y_pred - y_pred + y_actual - y_actual, tf.float32)

    def get_config(self):
        return {}


register_custom_object("NoLoss", NoLoss)
