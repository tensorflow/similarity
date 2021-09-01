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

import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow_similarity.types import IntTensor


def confusion_matrix(y_pred: IntTensor,
                     y_true: IntTensor,
                     normalize: bool = True,
                     labels: IntTensor = None,
                     title: str = 'Confusion matrix',
                     cmap: str = 'Blues'):
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
    """
    # Ensure we are working with integer tensors.
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype='int32')
    y_true = tf.cast(tf.convert_to_tensor(y_true), dtype='int32')

    cm = tf.math.confusion_matrix(y_true, y_pred)
    cm = tf.cast(cm, dtype='float')
    accuracy = tf.linalg.trace(cm) / tf.math.reduce_sum(cm)
    misclass = 1 - accuracy

    if normalize:
        cm = tf.math.divide_no_nan(
                cm,
                tf.math.reduce_sum(cm, axis=1)[:, np.newaxis]
        )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

    cm_max = tf.math.reduce_max(cm)
    thresh = cm_max / 1.5 if normalize else cm_max / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        color = "white" if val > thresh else "black"
        txt = "%.2f" % val if val > 0.0 else "0"
        plt.text(j, i, txt, horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()
