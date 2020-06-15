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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import mock
from mock import patch

import collections
import os
import shutil

import tensorflow as tf
import numpy as np

from tensorflow_similarity.api.callbacks.plugins import ConfusionMatrixCallbackPlugin


# data type for neighbors returned by the database
LabeledNeighbor = collections.namedtuple("Neighbor",
                                         ["distance", "index", "label"])


class TestConfusionMatrixVisualizationPlugin(unittest.TestCase):
    """Tests Confusion Matrix Visualization Plugin.
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
            subfolder_name="confusion_matrix"):
        """Helper method to check if a tf.event is generated inside
            folder/subfolder directory.

        Args:
            folder_name (str, optional): Folder name to test.
                Defaults to "logs".
            subfolder_name (str, optional): Subfolder name to test
                within the folder.  Defaults to "confusion_matrix".
        """

        # check that the defaults log folders and files are created
        self.assertIn(folder_name, os.listdir('.'))
        self.assertIn(subfolder_name, os.listdir('./' + folder_name))

        confusion_matrix_folder_contents = os.listdir(
            './{}/{}'.format(folder_name, subfolder_name))

        # check that we only create one tfevent
        self.assertEqual(len(confusion_matrix_folder_contents), 1)

        content_name = confusion_matrix_folder_contents[0]
        self.assertIn('.tfevents', content_name)

    def testInitializationDefault(self):
        """Simple test to assert that the default attributes is set correctly
            for ConfusionMatrixCallbackPlugin.
        """

        expected_frequency = 1
        expected_title = 'Confusion Matrix'
        confusion_matrix_plugin = ConfusionMatrixCallbackPlugin()

        # check that the attributes for the confusion matrix plugin is
        # set correctly
        self.assertEqual(confusion_matrix_plugin.frequency, expected_frequency)
        self.assertEqual(confusion_matrix_plugin.title, expected_title)
        self.assertIsInstance(
            confusion_matrix_plugin.file_writer,
            tf.summary.SummaryWriter)

        # test that a tf.event object is genereated in default log folder
        self._test_tfevent_generated()

    def testInitializationWithFileWriter(self):
        """Simple test to assert we are able to initialize
            ConfusionMatrixCallbackPlugin with a file_writer as well.
        """

        folder_name = 'custom_logs'
        subfolder_name = "custom_subfolder"
        logdir = '{}/{}'.format(folder_name, subfolder_name)
        expected_file_writer = tf.summary.create_file_writer(logdir)
        confusion_matrix_plugin = ConfusionMatrixCallbackPlugin(
            file_writer=expected_file_writer)

        # check that the file writer we created is set correctly
        self.assertEqual(
            confusion_matrix_plugin.file_writer,
            expected_file_writer)

        # test that a tf.event object is genereated in default log folder
        self._test_tfevent_generated(folder_name, subfolder_name)

        # complete the test, remove the directory created
        shutil.rmtree('{}/{}'.format(self.current_dir, folder_name))

    def testConfusionMatrixTensor(self):
        """Simple test to assert that the internal method,
            _confusion_matrix_tensor, behaves as expected for
            different size of confusion matrix.
        """

        # test small confusion matrix return expected size tensor
        expected_small_tensor_dimension = [1, 800, 800, 4]
        small_confusion_matrix = np.random.randint(5, size=(2, 2))
        small_class_name = ['google', 'android']
        confusion_matrix_plugin = ConfusionMatrixCallbackPlugin()

        small_tensor = confusion_matrix_plugin._confusion_matrix_tensor(
            small_confusion_matrix, small_class_name)

        small_tensor_dimensions = small_tensor.get_shape().as_list()

        self.assertIsInstance(small_tensor, tf.Tensor)
        self.assertListEqual(
            small_tensor_dimensions,
            expected_small_tensor_dimension)

        # test large confusion matrix return expected larger size tensor
        expected_large_tensor_dimension = [1, 1000, 1000, 4]
        large_confusion_matrix = np.random.randint(5, size=(10, 10))
        large_class_name = ['{}'.format(i) for i in range(10)]

        large_tensor = confusion_matrix_plugin._confusion_matrix_tensor(
            large_confusion_matrix, large_class_name)

        large_tensor_dimensions = large_tensor.get_shape().as_list()

        self.assertIsInstance(large_tensor, tf.Tensor)
        self.assertListEqual(
            large_tensor_dimensions,
            expected_large_tensor_dimension)

        # test that when the number of classes in class_names does not agree
        # with the dimension of confusion matrix an exception is raised
        with self.assertRaises(Exception):
            confusion_matrix_plugin._confusion_matrix_tensor(
                small_confusion_matrix, large_class_name)

    def testComputeMetrics(self):
        """Simple test to assert that compute_metrics for confusion matrix
            callback plugin is working as expected.
        """

        confusion_matrix_callback = ConfusionMatrixCallbackPlugin()
        mock_confusion_matrix_tensor = mock.MagicMock()
        mock_confusion_matrix_tensor.return_value = tf.zeros((1, 2, 2, 4))
        confusion_matrix_callback._confusion_matrix_tensor = \
            mock_confusion_matrix_tensor

        confusion_matrix_callback.compute_metrics(
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

        mock_confusion_matrix_tensor.assert_called_once()

        # test that a tf.event object is genereated in default log folder
        self._test_tfevent_generated()


if __name__ == '__main__':
    unittest.main()
