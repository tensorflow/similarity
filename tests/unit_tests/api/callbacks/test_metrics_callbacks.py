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
"""This file contains unit test for moirai/api/callbacks/metrics_callbacks.py.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import mock

from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallbackPlugin
from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallback


class TestMetricsCallback(unittest.TestCase):

    def setUp(self):

        # define expectation
        self.expected_embedding_target = [[0, 1], [2, 3]]
        self.expected_embedding_test = [[0, 3], [4, 3]]
        self.expected_neighbors = [0, 1]
        self.expected_epoch = 2
        self.expected_logs = {"training_loss": 0.5}

        # define simple test and target data
        self.x_test = {"example": [[1, 2], [2, 3]]}
        self.y_test = [1, 2]
        self.x_targets = {"example": [[0, 1], [3, 2]]}
        self.y_targets = [1, 2]

        # set up mock plugins
        num_plugins = 3
        mock_plugins = [None] * num_plugins
        for i in range(num_plugins):
            mock_plugin = MetricsCallbackPlugin()
            mock_plugin.compute_metrics = mock.MagicMock()
            mock_plugins[i] = mock_plugin

        self.mock_plugins = mock_plugins

        # set up mock simhash
        mock_simhash = mock.MagicMock()

        # set up mock database for mock simhash
        mock_database = mock.MagicMock()
        mock_database.get_embeddings.return_value = self.expected_embedding_target
        mock_database.query.return_value = self.expected_neighbors

        mock_simhash.build_database.return_value = mock_database
        mock_simhash.predict.return_value = self.expected_embedding_test

        self.mock_database = mock_database
        self.mock_simhash = mock_simhash

    def testMetricsPluginDefaultFrequency(self):
        """Simple test to assert that the default frequency for
            Metrics Plugin is 1.
        """
        default_plugin = MetricsCallbackPlugin()
        self.assertEqual(
            default_plugin.frequency,
            1,
            "The default frequency of plugin is {} instead of 1".format(
                default_plugin.frequency))

    def testMetricsPluginAssertion(self):
        """Simple test to assert that the correct assertion is thrown
            when we initialize the plugin with invalid frequency.
        """

        with self.assertRaises(Exception):
            MetricsCallbackPlugin(frequency=0.5)

        with self.assertRaises(Exception):
            MetricsCallbackPlugin(frequency=2.5)

    def testMetricsCallback(self):
        """Simple test case that assert that will call each plugin's
            compute metrics and that all computation on generating target
            and test embeddings are only been called once.
        """

        metrics_callback = MetricsCallback(
            self.mock_plugins,
            self.x_test,
            self.y_test,
            self.x_targets,
            self.y_targets)

        metrics_callback.set_simhash(self.mock_simhash)
        metrics_callback.on_epoch_end(self.expected_epoch, self.expected_logs)

        # test that each plugins have been called once with expected arguments
        for plugin in self.mock_plugins:
            plugin.compute_metrics.assert_called_once_with(
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

        # test that even when we have multiple plugins the common computations
        # are only been called once each.
        self.mock_simhash.build_database.assert_called_once_with(
            self.x_targets, self.y_targets)
        self.mock_simhash.predict.assert_called_once_with(self.x_test)
        self.mock_database.get_embeddings.assert_called_once()
        self.mock_database.query.assert_called_once_with(
            self.expected_embedding_test, N=len(self.y_targets))

    def testMetricsCallbackWithDifferentFrequencies(self):
        """Tests that Metrics Callback handles plugins with different
            frequencies well.
        """

        # number of epoch we want to test
        num_epochs = 100

        # build a mock plugin with custom frequency
        custom_frequency = 10
        infrequent_mock_plugin = MetricsCallbackPlugin(
            frequency=custom_frequency)
        infrequent_mock_plugin.compute_metrics = mock.MagicMock()

        # add the custom mock plugin to the list of mock plugins
        all_mock_plugins = self.mock_plugins + [infrequent_mock_plugin]

        metrics_callback = MetricsCallback(
            all_mock_plugins,
            self.x_test,
            self.y_test,
            self.x_targets,
            self.y_targets)

        metrics_callback.set_simhash(self.mock_simhash)

        for epoch in range(num_epochs):
            metrics_callback.on_epoch_end(epoch, self.expected_logs)

            # test that each plugins have been called once with expected
            # arguments.
            for plugin in self.mock_plugins:
                plugin.compute_metrics.assert_called_once_with(
                    self.mock_simhash,
                    self.mock_database,
                    self.expected_neighbors,
                    self.expected_embedding_test,
                    self.x_test,
                    self.y_test,
                    self.expected_embedding_target,
                    self.x_targets,
                    self.y_targets,
                    epoch,
                    self.expected_logs)

            # test that the plugin with custom frequency is only called
            # once every set frequency.
            if epoch % custom_frequency == 0:
                infrequent_mock_plugin.compute_metrics.assert_called_once_with(
                    self.mock_simhash,
                    self.mock_database,
                    self.expected_neighbors,
                    self.expected_embedding_test,
                    self.x_test,
                    self.y_test,
                    self.expected_embedding_target,
                    self.x_targets,
                    self.y_targets,
                    epoch,
                    self.expected_logs)
            else:
                self.assertFalse(infrequent_mock_plugin.compute_metrics.called)

            # test that even when we have multiple plugins the common computations
            # are only been called once each.
            self.mock_simhash.build_database.assert_called_once_with(
                self.x_targets, self.y_targets)
            self.mock_simhash.predict.assert_called_once_with(self.x_test)
            self.mock_database.get_embeddings.assert_called_once()
            self.mock_database.query.assert_called_once_with(
                self.expected_embedding_test, N=len(self.y_targets))

            # reset mocks for next epoch
            for plugin in self.mock_plugins:
                plugin.compute_metrics.reset_mock()
            self.mock_simhash.reset_mock()
            self.mock_database.reset_mock()
            infrequent_mock_plugin.compute_metrics.reset_mock()


if __name__ == '__main__':
    unittest.main()
