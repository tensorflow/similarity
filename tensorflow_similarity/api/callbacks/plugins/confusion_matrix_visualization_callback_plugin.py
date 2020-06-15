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

"""This file contains the plugin for the plugin that will save the
        confusion matrix in Tensorboard throughout the training process.
"""

import tensorflow as tf

import itertools

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import numpy as np

from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallbackPlugin
from tensorflow_similarity.api.callbacks.plugins.utils.plugin_utils import plot_to_tensor


class ConfusionMatrixCallbackPlugin(MetricsCallbackPlugin):
    """A plugin that computes confusion matrix and save it via Tensorboard.

    Arguments:
        file_writer (String|FileWriter):
            String: the path to the directory where we
                want to log our confusion matrix.
            FileWriter: the FileWriter object we want to write to.
        title (String): Optional, the title for the confusion matrix.
        frequency (int): Optional, frequency (in epochs) at which
            compute_metrics will be performed. Default to 1.
    """

    def __init__(
            self,
            file_writer='logs/confusion_matrix',
            title='Confusion Matrix',
            frequency=1):

        super(ConfusionMatrixCallbackPlugin, self).__init__(frequency)

        self.title = title
        self.tensor_name = title + '/image'

        if isinstance(file_writer, str):
            self.file_writer = tf.summary.create_file_writer(file_writer)
        else:
            self.file_writer = file_writer

    def compute_metrics(self,
                        simhash,
                        database,
                        neighbors,
                        embedding_test,
                        x_test,
                        y_test,
                        embedding_targets,
                        x_targets,
                        y_targets,
                        epoch,
                        logs={}):
        """Overwritten from parent class, this method computes the confusion matrix.

        Arguments:
            simhash (SimHashInterface): SimHash object,
                used for training/inference.
            database (dict: validation set -> Database): Per-dataset database
                (searchable mapping of embedding -> label) of targets.
            neighbors (dict: validation set -> list[LabeledNeighbor]):
                Per-dataset, per-example exhaustive lists of nearest neighbors.
            embedding_test (dict: validation set -> list[embeddings]):
                Per-dataset test embeddings.
            x_test (dict: validation set -> list[???]):
                Per-dataset test examples.
            y_test (dict: validation set -> list[String|integer]):
                Per-dataset test labels, the label can either be strings or
                integers, and those will be used as the labels for the
                confusion matrix.
            embedding_targets (dict: validation set -> list[embeddings]):
                Per-dataset target embeddings.
            x_targets (dict: validation set -> list[???]):
                Per-dataset target examples.
            y_targets (dict: validation set -> list[String|integer]):
                Per-dataset target labels, the label can either be strings or
                integers, and those will be used as the labels for the
                confusion matrix.
            epoch (int): Current epoch, from the Keras callback.

        Keyword Arguments:
            logs (dict): Current logs, from the Tensorflow Keras callback.
        """

        # The current implmentation states that the predicted label
        # is the closest target to a given datapoint
        predicted_labels = [neighbor[0].label for neighbor in neighbors]

        # In N-way k-shot learning we will have k targets per class, we
        # just need the unique ones, this simple approach also preserves order.
        unique_target_labels = list(set(y_targets))

        confusion_matrix_data = confusion_matrix(
            y_test, predicted_labels, unique_target_labels)
        confusion_matrix_tensor = self._confusion_matrix_tensor(
            confusion_matrix_data, unique_target_labels)

        with self.file_writer.as_default():
            tf.summary.image(
                self.tensor_name,
                confusion_matrix_tensor,
                step=epoch)

    def _confusion_matrix_tensor(self, confusion_matrix, class_names):
        """Returns a tensor containing the plotted confusion matrix.

        Arguments:
            confusion_matrix (array, shape = [n, n]): a confusion matrix of
                integer classes.
            class_names (array, shape = [n]): String names of the
                integer classes.

        Returns:
            confusion_matrix_tensor: a tensor of shape
                (1, height, width, channels) that contains the
                confusion matrix plot.
        """

        assert len(confusion_matrix) == len(class_names), \
            'Number of classes does not agree with the dimension of the confusion matrix.'

        # convert confusion matrix to numpy array
        confusion_matrix = np.asarray(confusion_matrix)

        # scale the size of the plot by the number of classes
        size = max(len(class_names), 8)

        figure = plt.figure(figsize=(size, size))
        plt.imshow(
            confusion_matrix,
            interpolation='nearest',
            cmap=plt.get_cmap('Blues'))
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        epsilon = 10 ** -10
        row_sum = confusion_matrix.sum(axis=1)[:, np.newaxis] + epsilon
        confusion_matrix = np.around(
            confusion_matrix.astype('float') / row_sum, decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = confusion_matrix.max() / 2.0
        for i, j in itertools.product(
            range(
                confusion_matrix.shape[0]), range(
                confusion_matrix.shape[1])):
            color = "white" if confusion_matrix[i, j] > threshold else "black"
            plt.text(j, i, confusion_matrix[i, j],
                     horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        confusion_matrix_tensor = plot_to_tensor(figure)
        return confusion_matrix_tensor
