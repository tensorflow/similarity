# Lint as: python3
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
"""This file contains tests for confusion matrix visualization callback plugin.
"""

from __future__ import absolute_import, division, print_function

import os
import shutil
import unittest

import cv2
import mock
import numpy as np
import tensorflow as tf
from mock import patch
from tensorflow_similarity.api.callbacks.plugins import EmbeddingProjectorCallbackPlugin
from tensorflow_similarity.api.engine.database import LabeledNeighbor


class TestEmbeddingProjectorVisualizationPlugin(unittest.TestCase):
    """Tests Embedding Projector Visualization Plugin.
    """

    def setUp(self):
        self.current_dir = os.getcwd()

        # removing logs directory if it exists
        if 'logs' in os.listdir('.'):
            shutil.rmtree(self.current_dir + '/logs')

        # define expectations
        self.expected_embedding_target = [[0, 1], [2, 3]]
        self.expected_embedding_test = [[0, 3], [4, 3]]

        self.expected_neighbors = []

        for _ in range(len(self.expected_embedding_test)):
            closest_neighbor = LabeledNeighbor(index=0, distance=0.32, label=1)
            second_neighbor = LabeledNeighbor(index=1, distance=0.52, label=2)
            neighbors = [closest_neighbor, second_neighbor]
            self.expected_neighbors.append(neighbors)

        self.expected_epoch = 2
        self.expected_logs = {"training_loss": 0.5}

        # define simple test and target data
        self.x_test = {"example": [[1, 2], [2, 3]]}
        self.y_test = [1, 2]
        self.x_targets = {"example": [[0, 1], [3, 2]]}
        self.y_targets = [1, 2]

        # read in example image datas
        black_and_white_image = cv2.imread("./images/black_and_white.jpeg", 0)
        self.black_and_white_data = np.array([black_and_white_image])
        color_image = cv2.imread("./images/color.jpeg")
        self.color_data = np.array([color_image])

        # set up mock simhash
        mock_simhash = mock.MagicMock()

        # set up mock database for mock simhash
        mock_database = mock.MagicMock()
        mock_database.get_embeddings.return_value = self.expected_embedding_target
        mock_database.neighbors.return_value = self.expected_neighbors

        mock_simhash.build_database.return_value = mock_database
        mock_simhash.predict.return_value = self.expected_embedding_test

        self.mock_database = mock_database
        self.mock_simhash = mock_simhash

    def tearDown(self):
        # removing logs directory if it exists
        if 'logs' in os.listdir('.'):
            shutil.rmtree(self.current_dir + '/logs')

    def _test_tfevent_generated(
            self,
            folder_name="logs",
            subfolder_name="projector"):
        """Helper method to check if a tf.event is generated inside
            folder/subfolder directory.

        Args:
            folder_name (str, optional): Folder name to test.
                Defaults to "logs".
            subfolder_name (str, optional): Subfolder name to test
                within the folder.  Defaults to "projector".
        """

        # check that the defaults log folders and files are created
        self.assertIn(folder_name, os.listdir('.'))
        self.assertIn(subfolder_name, os.listdir('./' + folder_name))

        projector_folder_contents = os.listdir(
            './{}/{}'.format(folder_name, subfolder_name))

        # check that we only create one tfevent
        self.assertEqual(len(projector_folder_contents), 1)

        content_name = projector_folder_contents[0]
        self.assertIn('.tfevents', content_name)

    def testInitializationDefault(self):
        """Simple test to assert that the default attributes is set correctly
            for ConfusionMatrixCallbackPlugin.
        """

        expected_frequency = 1
        expected_log_dir = "logs/projector"
        expected_sprite_file_name = "sprite.png"
        expected_meta_file_name = "metadata.tsv"
        expected_test_embeddings_tensors = dict()

        embedding_projector_plugin = EmbeddingProjectorCallbackPlugin()

        # check that the attributes for the confusion matrix plugin is
        # set correctly
        self.assertEqual(
            embedding_projector_plugin.frequency,
            expected_frequency)
        self.assertEqual(
            embedding_projector_plugin.sprite_file_name,
            expected_sprite_file_name)
        self.assertEqual(
            embedding_projector_plugin.meta_file_name,
            expected_meta_file_name)
        self.assertEqual(embedding_projector_plugin.log_dir, expected_log_dir)
        self.assertDictEqual(
            embedding_projector_plugin.test_embeddings_tensors,
            expected_test_embeddings_tensors)

        # check that those attributes are set to None
        self.assertIsNone(embedding_projector_plugin.sprite_image_key)
        self.assertIsNone(embedding_projector_plugin.x_test)
        self.assertIsNone(embedding_projector_plugin.y_test)
        self.assertIsNone(embedding_projector_plugin.sprite_image)

    def testIsGrayscale(self):
        """Simple test to assert that the internal method, is_gray_scale.
        """

        embedding_projector_plugin = EmbeddingProjectorCallbackPlugin()

        self.assertTrue(
            embedding_projector_plugin._is_grayscale(
                self.black_and_white_data))
        self.assertFalse(
            embedding_projector_plugin._is_grayscale(
                self.color_data))

    def testImagesToSprite(self):
        """Simple test to assert that the internal method,
            _images_to_sprite, behaves as expected.
        """

        embedding_projector_plugin = EmbeddingProjectorCallbackPlugin()

        color_sprite_image = embedding_projector_plugin._images_to_sprite(
            self.color_data)
        self.assertEqual(color_sprite_image.shape, (375, 500, 3))

        black_and_white_sprite_image = embedding_projector_plugin._images_to_sprite(
            self.black_and_white_data)
        self.assertEqual(black_and_white_sprite_image.shape, (333, 500, 3))

    def testComputeMetrics(self):
        """Simple test to assert that compute_metrics for embedding projector
            callback plugin is working as expected.
        """

        embedding_projector_callback = EmbeddingProjectorCallbackPlugin()

        expected_keys = [
            "epoch_{0}_embeddings".format(
                self.expected_epoch + 1)]

        embedding_projector_callback.compute_metrics(
            self.mock_simhash,
            self.mock_database,
            self.expected_neighbors,
            self.expected_embedding_test,
            self.x_test,
            self.y_test,
            self.expected_embedding_target,
            self.x_targets,
            self.y_targets,
            self.expected_epoch,
            self.expected_logs)

        self.assertEqual(
            len(embedding_projector_callback.test_embeddings_tensors), 1)
        actual_keys = list(
            embedding_projector_callback.test_embeddings_tensors.keys())
        self.assertListEqual(
            actual_keys,
            expected_keys)


if __name__ == '__main__':
    unittest.main()
