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

"""This file contains unit test for
    moirai/api/callbacks/plugins/closest_item_visualization_callback_plugin
    .py
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
from tensorflow_similarity.api.callbacks.plugins import ClosestItemsCallbackPlugin
from tensorflow_similarity.api.engine.database import LabeledNeighbor


class TestClosestItemsVisualizationPlugin(unittest.TestCase):
    """Tests Closest Items Visualization Plugin."""

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
        self.y_test = [1, 2]
        self.y_targets = [1, 2]

        # read in test image data from /image folder
        test_image_path_root = "./images/black_and_white_image_"
        extension = ".png"
        self.x_test = {"image": []}
        self.x_targets = {"image": []}

        for i in range(1, 5):
            path = test_image_path_root + str(i) + extension
            image = cv2.imread(path, 0)

            # put the first 2 images in test and other 2 in targets
            if i <= 2:
                self.x_test["image"].append(image)
            else:
                self.x_targets["image"].append(image)

        self.similiar_indices_matrix = [[0, 1], [1, 0]]
        self.distances = [[0.23, 0.42], [0.01, 0.37]]

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
            subfolder_name="closest_items"):
        """Helper method to check if a tf.event is generated inside
            folder/subfolder directory.

        Args:
            folder_name (str, optional): Folder name to test.
                Defaults to "logs".
            subfolder_name (str, optional): Subfolder name to test
                within the folder.  Defaults to "closest_items".
        """

        # check that the defaults log folders and files are created
        self.assertIn(folder_name, os.listdir('.'))
        self.assertIn(subfolder_name, os.listdir('./' + folder_name))

        folder_contents = os.listdir(
            './{}/{}'.format(folder_name, subfolder_name))

        # check that we only create one tfevent
        self.assertEqual(len(folder_contents), 1)

        content_name = folder_contents[0]
        self.assertIn('.tfevents', content_name)

    def testInitializationDefault(self):
        """Simple test to assert that the default attributes is set correctly
            for ClosestItemsCallbackPlugin.
        """

        expected_frequency = 1
        expected_N = 5
        expected_title = 'Closest Items'
        expected_image_key = 'image'
        closest_items_plugin = ClosestItemsCallbackPlugin()

        # check that the attributes for the confusion matrix plugin is
        # set correctly.
        self.assertEqual(closest_items_plugin.frequency, expected_frequency)
        self.assertEqual(closest_items_plugin.title, expected_title)
        self.assertEqual(closest_items_plugin.N, expected_N)
        self.assertEqual(closest_items_plugin.image_key, expected_image_key)
        self.assertIsInstance(
            closest_items_plugin.file_writer,
            tf.summary.SummaryWriter)
        self.assertTrue(closest_items_plugin.show_unique_targets)

        # test that a tf.event object is genereated in default log folder
        self._test_tfevent_generated()

    def testClosestItemGridTensor(self):
        """Simple test to assert that the internal method,
            _closest_item_tensor, behaves as expected.
        """

        closest_items_plugin = ClosestItemsCallbackPlugin(
            N=2, image_key="image")

        closest_items_tensor = closest_items_plugin._closest_item_grid_tensor(
            self.expected_neighbors,
            self.x_test,
            self.x_targets,
            self.y_targets)

        # test that the returned object is a tensorflow Tensor
        self.assertIsInstance(closest_items_tensor, tf.Tensor)

        # test that a tf.event object is genereated in default log folder
        self._test_tfevent_generated()

    def testComputeMetrics(self):
        """Simple test to assert that compute_metrics for closest items
            callback plugin is working as expected.
        """

        closest_items_callback = ClosestItemsCallbackPlugin(N=2)

        # mock _closest_item_grid_tensor method
        mock_closest_item_grid_tensor = mock.MagicMock()
        mock_closest_item_grid_tensor.return_value = tf.zeros((1, 2, 2, 4))
        closest_items_callback._closest_item_grid_tensor = \
            mock_closest_item_grid_tensor

        closest_items_callback.compute_metrics(
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

        mock_closest_item_grid_tensor.assert_called_once()

        # test that a tf.event object is genereated in default log folder
        self._test_tfevent_generated()


if __name__ == '__main__':
    unittest.main()
