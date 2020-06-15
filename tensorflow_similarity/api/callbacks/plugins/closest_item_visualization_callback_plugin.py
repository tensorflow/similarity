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

"""This file contains the plugin for the plugin that will save
    closest items in Tensorboard throughout the training process.
"""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallbackPlugin
from tensorflow_similarity.api.callbacks.plugins.utils.plugin_utils import plot_to_tensor
from tensorflow_similarity.api.engine.database import Database


class ClosestItemsCallbackPlugin(MetricsCallbackPlugin):
    """A metric plugin that computes closest items and save it via
        Tensorboard.

    Arguments:
        log_dir (String): The path to the directory where we
            want to log our confusion matrix.
        title (String): Optional, the title for the confusion matrix.
        frequency (int): Optional, frequency (in epochs) at which
            compute_metrics will be performed. Default to 1.
        image_key (String): The key (feature name) within the validation
            data dictionary that contains the image data. Defaults to 'image'.
        N (int): Optional, the number of closest items shown per
            target input. Defaults to 5.
        show_unique_targets (boolean): Optional, show only one
            target per class in the target dataset if True. False to show all
            targets specified in the target dataset. Useful when doing single
            shot learning when we have multiple targets in the same class but
            only interest to see one target per class. Defaults to True.
    """

    def __init__(
            self,
            log_dir='logs/closest_items',
            title='Closest Items',
            frequency=1,
            image_key="image",
            N=5,
            show_unique_targets=True):

        super(ClosestItemsCallbackPlugin, self).__init__(frequency)

        self.title = title
        self.tensor_name = title + '/image'
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.N = N
        self.image_key = image_key
        self.show_unique_targets = show_unique_targets

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
            y_test (dict: validation set -> list[int]):
                Per-dataset test labels.
            embedding_targets (dict: validation set -> list[embeddings]):
                Per-dataset target embeddings.
            x_targets (dict: validation set -> list[???]):
                Per-dataset target examples.
            y_targets (dict: validation set -> list[int]):
                Per-dataset target labels.
            epoch (int): Current epoch, from the Keras callback.

        Keyword Arguments:
            logs (dict): Current logs, from the Tensorflow Keras callback.
        """

        # make sure that N is at most the number of examples in the test set.
        self.N = min(self.N, len(y_test))

        # TODO(b/142810605): currently we are building this test_database
        # (database with all test points) as this is the only callback plugin
        # that needs it, this will be updated once the training loop rewrite
        # is completed so we can just fetch it instead of compute it.
        test_database = simhash.build_database(x_test, y_test)
        target_neighbors = test_database.query(embedding_targets, N=self.N)

        # filter out unique targets in the target dataset if specified to show
        # only one target per class
        if self.show_unique_targets:
            seen = set()
            unique_target_neighbors = []
            unique_y_targets = []
            unique_x_targets = dict()
            unique_x_targets[self.image_key] = []
            for x, y, neighbors in zip(
                    x_targets[self.image_key], y_targets, target_neighbors):
                if y not in seen:
                    seen.add(y)
                    unique_y_targets.append(y)
                    unique_x_targets[self.image_key].append(x)
                    unique_target_neighbors.append(neighbors)

            x_targets = unique_x_targets
            y_targets = unique_y_targets
            target_neighbors = unique_target_neighbors

        closest_items_tensor = self._closest_item_grid_tensor(
            target_neighbors,
            x_test,
            x_targets,
            y_targets)

        with self.file_writer.as_default():
            tf.summary.image(
                self.tensor_name,
                closest_items_tensor,
                step=epoch)

    def _closest_item_grid_tensor(
            self,
            target_neighbors,
            x_test,
            x_targets,
            y_targets):
        """Plot the most similiar / closest items grid.

        Arguments:
            target_neighbors (list[list[LabeledNeighbors]]): target_neighbors
                is a 2-dimensional neighbors matrix, where
                target_neighbors[i][j] is the LabeledNeighbor object (defined
                in database.py) that has 3 attributes -- distance, index, label.
            x_test (dict: validation set -> list[???]):
                Per-dataset test examples.
            embedding_targets (dict: validation set -> list[embeddings]):
                Per-dataset target embeddings.
            x_targets (dict: validation set -> list[???]):
                Per-dataset target examples.
            y_targets (dict: validation set -> list[int]):
                Per-dataset target labels.

        Returns:
            closest_items_tensor: a tensor of shape
                (1, height, width, channels) that contains the
                closest items plot.
        """

        num_row = len(y_targets)
        # add 2 for one additonal column for target image and
        # another for divider column.
        num_col = self.N + 2

        # scale the size of the plot by factor of 2 for better
        # looking display.
        scale = 2
        plt_width = num_col * scale
        plt_height = num_row * scale

        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(plt_width, plt_height))

        # TODO(b/142677389): The current implmentation can only handles image
        # data (when target_images[i] is an image), will update
        # the method for other types of data (text, etc) in later PR.
        target_images = x_targets[self.image_key]
        test_images = x_test[self.image_key]

        for row_id in range(num_row):
            for col_id in range(num_col):

                # the "divider" column, leave blank
                if col_id == 1:
                    continue

                # the subplot id that we want to draw, it is 1-indexed
                subplot_id = row_id * num_col + col_id + 1

                ax = plt.subplot(num_row, num_col, subplot_id)

                # remove ticks and grid
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)

                # display target image for this column
                if col_id == 0:
                    target_image = target_images[row_id]
                    plt.imshow(target_image, plt.get_cmap('binary'))

                    # set the label/title
                    target_label = y_targets[row_id]
                    ax.title.set_text(target_label)
                else:
                    # the rank of the closest image we want to display
                    # (e.g) when rank == 0 we want to display the closest image
                    rank = col_id - 2

                    # the LabeledNeighbor object to draw in this subplot
                    neighbor = target_neighbors[row_id][rank]

                    # draw test image
                    test_id = neighbor.index
                    test_image = test_images[test_id]
                    plt.imshow(test_image, plt.get_cmap('binary'))

                    # set the label/title
                    test_label = neighbor.label
                    ax.title.set_text(test_label)

                    # add distance as the x label
                    distance = neighbor.distance
                    distance_message = "{:4f}".format(distance)

                    # only add the word "distance" on the closest images
                    if rank == 0:
                        distance_message = "distance: " + distance_message

                    ax.set_xlabel(distance_message)

        # preventing overlaps
        plt.tight_layout()

        closest_items_tensor = plot_to_tensor(figure)
        return closest_items_tensor
