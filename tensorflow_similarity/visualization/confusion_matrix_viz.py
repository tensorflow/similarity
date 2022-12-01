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
"""Visualization utilities for TensorFlow Similarity."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow_similarity.types import FloatTensor, IntTensor


def confusion_matrix(
    y_pred: IntTensor,
    y_true: IntTensor,
    normalize: bool = True,
    labels: IntTensor | None = None,
    title: str = "Confusion matrix",
    cmap: str = "Blues",
    show: bool = True,
) -> tuple[Any, FloatTensor]:
    """Plot confusion matrix

    Args:
        y_pred: Model prediction returned by `model.match()`

        y_true: Expected class_id.

        normalize: Normalizes matrix values between 0 and 1.
        Defaults to True.

        labels: List of class string label to display instead of the class
        numerical ids. Defaults to None.

        title: Title of the confusion matrix. Defaults to 'Confusion matrix'.

        cmap: Color schema as CMAP. Defaults to 'Blues'.

        show: If the plot is going to be shown or not. Defaults to True.

    Returns:
        A Tuple containing the plot and confusion matrix.
    """

    with tf.device("/cpu:0"):
        # Ensure we are working with integer tensors.
        if not tf.is_tensor(y_pred):
            y_pred = tf.convert_to_tensor(np.array(y_pred))
        y_pred = tf.cast(y_pred, dtype="int32")
        if not tf.is_tensor(y_true):
            y_true = tf.convert_to_tensor(np.array(y_true))
        y_true = tf.cast(y_true, dtype="int32")

        cm = tf.math.confusion_matrix(y_true, y_pred)
        cm = tf.cast(cm, dtype="float")
        accuracy = tf.linalg.trace(cm) / tf.math.reduce_sum(cm)
        misclass = 1 - accuracy

        if normalize:
            cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])

        f, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title)
        f.colorbar(im)

        if labels is not None:
            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(labels)

        cm_max = tf.math.reduce_max(cm)
        thresh = cm_max / 1.5 if normalize else cm_max / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            txt = "%.2f" % val if val > 0.0 else "0"
            ax.text(j, i, txt, horizontalalignment="center", color=color)

        f.tight_layout()
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))

        if show:
            plt.show()

        return ax, cm
