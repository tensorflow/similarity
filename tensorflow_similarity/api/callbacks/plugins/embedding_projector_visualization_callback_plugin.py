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
    the embedding projector in Tensorboard throughout the training process.
"""

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallbackPlugin
from tensorflow_similarity.api.callbacks.plugins.utils.plugin_utils import plot_to_tensor
from tensorboard.plugins import projector


class EmbeddingProjectorCallbackPlugin(MetricsCallbackPlugin):
    """A metric plugin that computes closest items and save it via
        Tensorboard.

    Arguments:
        file_writer (String): The path to the directory where we
                want to log our confusion matrix.
        sprite_image_key (String): The key (feature name) within the
            validation data dictionary that contains the image data that
            we want to read in as sprite, if None then we don't add sprite
            to the embedding projector. Default to None.
        frequency (int): Optional, frequency (in epochs) at which
            compute_metrics will be performed. Default to 1.
    """

    def __init__(
            self,
            log_dir='logs/projector',
            sprite_image_key=None,
            frequency=1):

        super(EmbeddingProjectorCallbackPlugin, self).__init__(frequency)

        self.log_dir = log_dir
        self.sprite_image_key = sprite_image_key
        self.sprite_image = None
        self.sprite_file_name = "sprite.png"
        self.meta_file_name = "metadata.tsv"

        self.config = projector.ProjectorConfig()

        # storing those variables throughout training
        self.test_embeddings_tensors = dict()
        self.x_test = None
        self.y_test = None

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

        # turn the embedding into a tf.Variable
        name = "epoch_{0}_embeddings".format(epoch + 1)
        test_embeddings_tensor = tf.Variable(embedding_test, name=name)

        # store the tensor, we will save all the tensors throughout
        # training period when training ends.
        self.test_embeddings_tensors[name] = test_embeddings_tensor

        # TODO(b/142810702): When we are refactoring embedding projector
        # plugin into it's own keras callback this will move to
        # on_train|evaluate_begin
        if self.x_test is None:
            self.x_test = x_test
            self.y_test = y_test

    def on_train_end(self, logs=None):

        # Create a directory at log_dir if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)

        # write meta data for the projector (labels)
        meta_path = os.path.join(self.log_dir, self.meta_file_name)

        with open(meta_path, 'w+') as f:
            for i in range(len(self.y_test)):
                label = self.y_test[i]
                f.write(str(label) + "\n")

        # create and save sprite image if user specify the key into
        # the test dataset for the image data.
        if self.sprite_image_key:
            sprite_images_data = self.x_test[self.sprite_image_key]
            self.sprite_image = self._images_to_sprite(sprite_images_data)
            sprite_image_path = os.path.join(
                self.log_dir, self.sprite_file_name)
            plt.imsave(sprite_image_path, self.sprite_image)

        check_point_dir = os.path.join(self.log_dir, "model.ckpt")

        checkpoint = tf.train.Checkpoint(
            **self.test_embeddings_tensors)
        checkpoint.save(check_point_dir)

        # Write the embedding configuration for each embedding tensors saved.
        for i in range(len(self.test_embeddings_tensors)):
            epoch = i * self.frequency + 1

            # due to how the checkpoint format has changed for tf 2.0, the
            # tensor_name is been changed to some strange string
            # (exampe: embedding/.ATTRIBUTES/VARIABLE_VALUE)
            # this work-around solution is proposed in the Github issue here:
            # https://github.com/tensorflow/tensorboard/issues/2471
            tensor_name = "epoch_{0}_embeddings/.ATTRIBUTES/VARIABLE_VALUE".format(
                epoch)

            # add the embedding to the projector API
            embedding = self.config.embeddings.add()
            embedding.tensor_name = tensor_name
            embedding.metadata_path = self.meta_file_name

            # add the path to sprite image file to the embedding config file
            # if user specified sprite image key.
            if self.sprite_image_key:
                embedding.sprite.image_path = self.sprite_file_name
                width = sprite_images_data.shape[1]
                height = sprite_images_data.shape[2]
                embedding.sprite.single_image_dim.extend([width, height])

        # write the embedding configuration to the appropriate file in log_dir
        projector.visualize_embeddings(self.log_dir, self.config)

    def _is_grayscale(self, data):
        """Helper method to see if data is grayscale image or RGB.

        Arguments:
            data: NxHxW[x3] tensor containing the images.
        """

        # The data is Grayscale if it has shape of NxHxW[x1] and
        # is RGB if it has shape of NxHxWx3
        return len(data.shape) == 3 or data.shape[-1] == 1

    def _images_to_sprite(self, data):
        """Creates the sprite image along with any necessary padding.

        Arguments:
            data: NxHxW[x3] tensor containing the images.

        Returns:
            data: Properly shaped HxWx3 image with any necessary padding.
        """

        is_grayscale = self._is_grayscale(data)

        # Add another dimension to the data if it's grayscale.
        if is_grayscale:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

        # normalize the image data.
        data = data.astype(np.float32)
        lowest = data.min(axis=(1, 2, 3), keepdims=True)
        highest = data.max(axis=(1, 2, 3), keepdims=True)
        diff = highest - lowest
        epilson = 10 ** -8

        # only normalize the data if there is any difference in pixels,
        # otherwise do not modify the data.
        if any(element > epilson for element in diff):
            data = (data - lowest) / diff

        # invert the color if the data is grayscale for better visual effect
        # as sprite to our embedding projector.
        if is_grayscale:
            data = 1 - data

        # pad the image data to make it a thumbnails image data.
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)

        # tile the individual thumbnails into an image.
        data = data.reshape(
            (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape(
            (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)

        return data
