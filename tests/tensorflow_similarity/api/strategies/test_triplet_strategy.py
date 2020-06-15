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
"""This file contains unit test for moirai/api/strategy/triplet_strategy.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import mock
from mock import patch
from mock import ANY

from tensorflow.keras.optimizers import Adam

from tensorflow_similarity.api.engine.similarity_model import SimilarityModel
from tensorflow_similarity.api.strategies.triplet_strategy import (
    HardTripletLossStrategy, TripletLossStrategy)
from tensorflow_similarity.api.tasks.utils_for_test import (
    gen_learnable_testdata, learnable_model)

# imports for patching
import tempfile


class TestTripletStrategy(unittest.TestCase):

    def setUp(self):

        # gather test data
        self.x, self.y = gen_learnable_testdata()
        self.tower_model = learnable_model()

        # define expectations
        self.expected_towers = ["anchor", "neg", "pos"]
        self.expected_tmp_dir = "./expected_tmp_dir"
        self.triplet_loss_strategy = "triplet_loss"
        self.hard_triplet_loss_strategy = "hard_triplet_loss"
        self.expected_default_optimizer_type = type(Adam(lr=0.001))
        self.x = [[0, 1], [0, 2]]
        self.y = [0, 1]

    def testTripletLossStrategyInitialization(self):
        """Simple test that asserts the correct default initialization of
                triplet loss strategy.
        """

        triplet_model = TripletLossStrategy(self.tower_model)

        # checking that all the attributes are set as expected
        self.assertEqual(triplet_model.tower_model, self.tower_model)
        self.assertEqual(triplet_model.hard_mining, False)
        self.assertIsInstance(
            triplet_model.raw_optimizer,
            self.expected_default_optimizer_type)
        self.assertIsNone(triplet_model.preprocessing)
        self.assertIsNone(triplet_model.augmentation)
        self.assertIsNone(triplet_model.database)
        self.assertListEqual(triplet_model.towers, self.expected_towers)
        self.assertEqual(triplet_model.strategy, self.triplet_loss_strategy)

    @patch.object(tempfile, 'mkdtemp', return_value="./expected_tmp_dir")
    def testHardTripletLossStrategyInitialization(self, mock_mkdtemp):
        """Simple test that asserts the correct default initialization of
                hard triplet loss strategy.
        """

        hard_triplet_model = HardTripletLossStrategy(self.tower_model)

        # checking that all the attributes are set as expected
        self.assertEqual(hard_triplet_model.tower_model, self.tower_model)
        self.assertEqual(hard_triplet_model.hard_mining, True)
        self.assertIsInstance(
            hard_triplet_model.raw_optimizer,
            self.expected_default_optimizer_type)
        self.assertIsNone(hard_triplet_model.preprocessing)
        self.assertIsNone(hard_triplet_model.augmentation)
        self.assertIsNone(hard_triplet_model.database)
        self.assertListEqual(hard_triplet_model.towers, self.expected_towers)
        self.assertEqual(
            hard_triplet_model.strategy,
            self.hard_triplet_loss_strategy)

        # assert that mkdtemp was called once for hard_mining_directory
        mock_mkdtemp.assert_called_once()

        self.assertEqual(
            hard_triplet_model.hard_mining_directory,
            self.expected_tmp_dir)

    @patch('moirai.api.strategies.triplet_strategy.TripletLossTask')
    @patch('moirai.api.strategies.triplet_strategy.MetaTask')
    def testTripletStrategyBuildTrainingTask(
            self, mock_meta_task, mock_triplet_loss_task):
        """Simple test that asserts the correct behavior when we invoked
                _build_training_task in TripletLossStrategy.
        """

        triplet_model = TripletLossStrategy(self.tower_model)

        mock_build_auxillary_tasks = mock.MagicMock()
        triplet_model._build_auxillary_tasks = mock_build_auxillary_tasks

        triplet_model._build_training_task(self.x, self.y)

        # assert calling with expected arugments to mocked methods
        mock_triplet_loss_task.assert_called_once_with(
            self.x,
            self.y,
            self.tower_model,
            augmentation=None,
            hard_mining=False,
            preprocessing=None)

        mock_build_auxillary_tasks.assert_called_once_with(self.x, self.y)

        mock_meta_task.assert_called_once_with(
            self.triplet_loss_strategy,
            self.tower_model,
            mock_triplet_loss_task.return_value,
            auxillary_tasks=mock_build_auxillary_tasks.return_value,
            optimizer=ANY)

    @patch('moirai.api.strategies.triplet_strategy.TripletLossTask')
    @patch('moirai.api.strategies.triplet_strategy.MetaTask')
    @patch.object(tempfile, 'mkdtemp', return_value="./expected_tmp_dir")
    def testHardTripletStrategyBuildTrainingTask(
            self, mock_mkdtemp, mock_meta_task, mock_triplet_loss_task):
        """Simple test that asserts the correct behavior when we invoked
                _build_training_task in HardTripletLossStrategy.
        """

        hard_triplet_model = HardTripletLossStrategy(self.tower_model)

        mock_build_auxillary_tasks = mock.MagicMock()
        hard_triplet_model._build_auxillary_tasks = mock_build_auxillary_tasks

        hard_triplet_model._build_training_task(self.x, self.y)

        # assert calling with expected arugments to mocked methods
        mock_triplet_loss_task.assert_called_once_with(
            self.x,
            self.y,
            self.tower_model,
            augmentation=None,
            hard_mining=True,
            hard_mining_directory="./expected_tmp_dir",
            preprocessing=None)

        mock_mkdtemp.assert_called_once()

        mock_build_auxillary_tasks.assert_called_once_with(self.x, self.y)

        mock_meta_task.assert_called_once_with(
            self.hard_triplet_loss_strategy,
            self.tower_model,
            mock_triplet_loss_task.return_value,
            auxillary_tasks=mock_build_auxillary_tasks.return_value,
            optimizer=ANY)


if __name__ == '__main__':
    unittest.main()
