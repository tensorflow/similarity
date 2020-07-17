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

# Lint as: python3
"""This file contains unit test for tensorflow_similarity/benchmark/result.py
"""

from __future__ import absolute_import, division, print_function

import os
import shutil
import unittest

import mock
import numpy as np
from tensorflow_similarity.benchmark import Result


class TestResult(unittest.TestCase):
    """Tests Result Class."""

    def _test_default_behavior(self, result):
        # assert both model_information and data_informaation are empty
        # if not provided by user
        self.assertFalse(result.model_information)
        self.assertFalse(result.data_information)

        # test training_information is set correctly
        self.assertEqual(result.training_information["num_epochs"], 0)
        self.assertEqual(result.training_information["inference_time"], 0.0)

        # assert epoch level metrics is set correctly
        expected_epoch_level_keys = (
            "accuracy",
            "v_score",
            "completeness",
            "homogeneity",
            "silhouette_score")
        actual_epoch_level_keys = tuple(result.epoch_level_metrics.keys())
        self.assertEqual(expected_epoch_level_keys, actual_epoch_level_keys)

        # assert best model metrics is set correctly
        expected_best_model_keys = (
            "accuracy",
            "epochs",
            "target_embeddings",
            "test_embeddings",
            "target_labels",
            "test_labels")
        actual_best_model_keys = tuple(result.best_model_metrics.keys())
        self.assertEqual(expected_best_model_keys, actual_best_model_keys)

    def testDefaultInitiailization(self):
        result = Result()

        self._test_default_behavior(result)

    def testToJsonRecoveryDefault(self):
        result = Result()
        json_str = result.to_json()
        recovered_result = Result.from_json(json_str)

        self._test_default_behavior(recovered_result)

    def testToJsonRecoveryModified(self):
        result = Result()

        expected_inference_time = 10.0
        expected_accuracy_list = [0.95, 0.99]
        expected_best_accuracy = 0.99
        key = "20_ways_1_shots"
        expected_target_embeddings = [0.123, 0.992, 0.9384, 3.4]

        result.training_information["inference_time"] = expected_inference_time
        result.epoch_level_metrics["accuracy"][key] = expected_accuracy_list
        result.best_model_metrics["accuracy"][key] = expected_best_accuracy
        result.best_model_metrics["target_embeddings"][key] = \
            expected_target_embeddings

        json_str = result.to_json()
        recovered_result = Result.from_json(json_str)

        actual_inference_time = \
            recovered_result.training_information["inference_time"]
        self.assertEqual(expected_inference_time, actual_inference_time)

        actual_accuracy_list = \
            recovered_result.epoch_level_metrics["accuracy"][key]
        self.assertListEqual(expected_accuracy_list, actual_accuracy_list)

        actual_best_accuracy = \
            recovered_result.best_model_metrics["accuracy"][key]
        self.assertEqual(expected_best_accuracy, actual_best_accuracy)

        actual_target_embeddings = \
            recovered_result.best_model_metrics["target_embeddings"][key]
        self.assertListEqual(
            expected_target_embeddings,
            actual_target_embeddings)


if __name__ == '__main__':
    unittest.main()
