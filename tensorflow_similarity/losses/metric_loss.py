# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Metric losses  base class."""

from typing import Any, Callable, Dict, Optional

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor
from tensorflow_similarity.utils import is_tensor_or_variable


class MetricLoss(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self, fn: Callable, reduction: Callable = tf.keras.losses.Reduction.AUTO, name: Optional[str] = None, **kwargs
    ):
        """Initializes `LossFunctionWrapper` class.
        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true: FloatTensor, y_pred: FloatTensor) -> FloatTensor:
        """Invokes the `LossFunctionWrapper` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Loss values per sample.
        """
        loss: FloatTensor = self.fn(y_true, y_pred, y_true, y_pred, **self._fn_kwargs)
        return loss

    def get_config(self) -> Dict[str, Any]:
        """Contains the loss configuration.

        Returns:
            A Python dict containing the configuration of the loss.
        """
        config = {}
        for k, v in iter(self._fn_kwargs.items()):
            if is_tensor_or_variable(v):
                config[k] = tf.keras.backend.eval(v)
            else:
                config[k] = v

        base_config = super().get_config()
        return {**base_config, **config}
