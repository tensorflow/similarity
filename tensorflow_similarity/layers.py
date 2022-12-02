# Copyright 2021 The TensorFlow Authors
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
"""Specialized Similarity `keras.layers`"""
from __future__ import annotations

import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils import conv_utils

from .types import FloatTensor, IntTensor


@tf.keras.utils.register_keras_serializable(package="Similarity")
class MetricEmbedding(layers.Dense):
    """L2 Normalized `Dense` layer.

    This layer is usually used as output layer, especially when using cosine
    distance as the similarity metric.
    """

    def call(self, inputs: FloatTensor) -> FloatTensor:
        x = super().call(inputs)
        normed_x: FloatTensor = tf.math.l2_normalize(x, axis=1)
        return normed_x


class GeneralizedMeanPooling(layers.Layer):
    def __init__(self, p: float = 3.0, data_format: str | None = None, keepdims: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.p = p
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.keepdims = keepdims

        if tf.math.abs(self.p) < 0.00001:
            self.compute_mean = self._geometric_mean
        elif self.p == math.inf:
            self.compute_mean = self._pos_inf
        elif self.p == -math.inf:
            self.compute_mean = self._neg_inf
        else:
            self.compute_mean = self._generalized_mean

    def compute_output_shape(self, input_shape: IntTensor) -> IntTensor:
        output_shape: IntTensor = self.gap.compute_output_shape(input_shape)
        return output_shape

    def call(self, inputs: FloatTensor) -> FloatTensor:
        raise NotImplementedError

    def _geometric_mean(self, x):
        x = tf.math.log(x)
        x = self.gap(x)
        return tf.math.exp(x)

    def _generalized_mean(self, x):
        x = tf.math.pow(x, self.p)
        x = self.gap(x)
        return tf.math.pow(x, 1.0 / self.p)

    def _pos_inf(self, x):
        raise NotImplementedError

    def _neg_inf(self, x):
        return self._pos_inf(x * -1) * -1

    def get_config(self) -> dict[str, Any]:
        config = {
            "p": self.p,
            "data_format": self.data_format,
            "keepdims": self.keepdims,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Similarity")
class GeneralizedMeanPooling1D(GeneralizedMeanPooling):
    r"""Computes the Generalized Mean of each channel in a tensor.

    $$
    \textbf{e} = \left[\left(\frac{1}{|\Omega|}\sum_{u\in{\Omega}}x^{p}_{cu}\right)^{\frac{1}{p}}\right]_{c=1,\cdots,C}
    $$

    The Generalized Mean (GeM) provides a parameter `p` that sets an exponent
    enabling the pooling to increase or decrease the contrast between salient
    features in the feature map.

    The pooling is equal to GlobalAveragePooling1D when `p` is 1.0 and equal
    to MaxPool1D when `p` is `inf`.

    This implementation shifts the feature map values such that the minimum
    value is equal to 1.0, then computes the mean pooling, and finally shifts
    the values back. This ensures that all values are positive as the
    generalized mean is only valid over positive real values.

    Args:
      p: Set the power of the mean. A value of 1.0 is equivalent to the
        arithmetic mean, while a value of `inf` is equivalent to MaxPool2D.
        Note, math.inf, -math.inf, and 0.0 are all supported, as well as most
        positive and negative values of `p`. However, large positive values for
        `p` may lead to overflow. In practice, math.inf should be used for any
        `p` larger than > 25.
      data_format: One of `channels_last` (default) or `channels_first`. The
        ordering of the dimensions in the inputs.  `channels_last`
        corresponds to inputs with shape `(batch, steps, features)` while
        `channels_first` corresponds to inputs with shape
        `(batch, features, steps)`.
      keepdims: A boolean, whether to keep the temporal dimension or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.  If `keepdims` is `True`, the temporal
        dimension are retained with length 1.  The behavior is the same as
        for `tf.reduce_max` or `np.max`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`
    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, features)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          3D tensor with shape `(batch_size, 1, features)`
        - If `data_format='channels_first'`:
          3D tensor with shape `(batch_size, features, 1)`
    """

    def __init__(self, p: float = 3.0, data_format: str | None = None, keepdims: bool = False, **kwargs) -> None:
        super().__init__(p=p, data_format=data_format, keepdims=keepdims, **kwargs)

        self.input_spec = layers.InputSpec(ndim=3)
        self.gap = layers.GlobalAveragePooling1D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.step_axis = 1 if self.data_format == "channels_last" else 2

    def call(self, inputs: FloatTensor) -> FloatTensor:
        x = inputs
        if self.data_format == "channels_last":
            mins = tf.math.reduce_min(x, axis=self.step_axis)
            x_offset = x - mins[:, tf.newaxis, :] + 1
            if self.keepdims:
                mins = mins[:, tf.newaxis, :]
        else:
            mins = tf.math.reduce_min(x, axis=self.step_axis)
            x_offset = x - mins[:, :, tf.newaxis] + 1
            if self.keepdims:
                mins = mins[:, :, tf.newaxis]

        x_offset = self.compute_mean(x_offset)
        x = x_offset + mins - 1

        return x

    def _pos_inf(self, x):
        mpl = layers.GlobalMaxPool1D(data_format=self.data_format)
        x = mpl(x)
        return x


@tf.keras.utils.register_keras_serializable(package="Similarity")
class GeneralizedMeanPooling2D(GeneralizedMeanPooling):
    r"""Computes the Generalized Mean of each channel in a tensor.

    $$
    \textbf{e} = \left[\left(\frac{1}{|\Omega|}\sum_{u\in{\Omega}}x^{p}_{cu}\right)^{\frac{1}{p}}\right]_{c=1,\cdots,C}
    $$

    The Generalized Mean (GeM) provides a parameter `p` that sets an exponent
    enabling the pooling to increase or decrease the contrast between salient
    features in the feature map.

    The pooling is equal to GlobalAveragePooling2D when `p` is 1.0 and equal
    to MaxPool2D when `p` is `inf`.

    This implementation shifts the feature map values such that the minimum
    value is equal to 1.0, then computes the mean pooling, and finally shifts
    the values back. This ensures that all values are positive as the
    generalized mean is only valid over positive real values.

    Args:
      p: Set the power of the mean. A value of 1.0 is equivalent to the
        arithmetic mean, while a value of `inf` is equivalent to MaxPool2D.
        Note, math.inf, -math.inf, and 0.0 are all supported, as well as most
        positive and negative values of `p`. However, large positive values for
        `p` may lead to overflow. In practice, math.inf should be used for any
        `p` larger than > 25.
      data_format: One of `channels_last` (default) or `channels_first`. The
        ordering of the dimensions in the inputs.  `channels_last`
        corresponds to inputs with shape `(batch, steps, features)` while
        `channels_first` corresponds to inputs with shape
        `(batch, features, steps)`.
      keepdims: A boolean, whether to keep the temporal dimension or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.  If `keepdims` is `True`, the temporal
        dimension are retained with length 1.  The behavior is the same as
        for `tf.reduce_max` or `np.max`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`
    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, features)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          3D tensor with shape `(batch_size, 1, features)`
        - If `data_format='channels_first'`:
          3D tensor with shape `(batch_size, features, 1)`
    """

    def __init__(self, p: float = 3.0, data_format: str | None = None, keepdims: bool = False, **kwargs) -> None:
        super().__init__(p=p, data_format=data_format, keepdims=keepdims, **kwargs)

        self.input_spec = layers.InputSpec(ndim=4)
        self.gap = layers.GlobalAveragePooling2D(data_format=data_format, keepdims=keepdims, **kwargs)

    def call(self, inputs: FloatTensor) -> FloatTensor:
        x = inputs
        if self.data_format == "channels_last":
            mins = tf.math.reduce_min(x, axis=[1, 2])
            x_offset = x - mins[:, tf.newaxis, tf.newaxis, :] + 1
            if self.keepdims:
                mins = mins[:, tf.newaxis, tf.newaxis, :]
        else:
            mins = tf.math.reduce_min(x, axis=[2, 3])
            x_offset = x - mins[:, :, tf.newaxis, tf.newaxis] + 1
            if self.keepdims:
                mins = mins[:, :, tf.newaxis, tf.newaxis]

        x_offset = self.compute_mean(x_offset)
        x = x_offset + mins - 1

        return x

    def _pos_inf(self, x):
        if self.data_format == "channels_last":
            pool_size = (x.shape[1], x.shape[2])
        else:
            pool_size = (x.shape[2], x.shape[3])
        mpl = layers.MaxPool2D(pool_size=pool_size, data_format=self.data_format)
        x = mpl(x)
        if not self.keepdims:
            if self.data_format == "channels_last":
                x = tf.reshape(x, (x.shape[0], x.shape[3]))
            else:
                x = tf.reshape(x, (x.shape[0], x.shape[1]))
        return x


class ActivationStdLoggingLayer(layers.Layer):
    """Computes the mean std of the activations of a layer.

    x = reduce_std(l2_normalize(inputs, axis=0), axis=-1)

    And then aggregate the per-batch mean of x over each epoch.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        x = tf.math.l2_normalize(inputs, axis=-1)
        x = tf.math.reduce_std(x, axis=0)
        self.add_metric(x, name=self.name, aggregation="mean")
        return inputs  # Pass-through layer.
