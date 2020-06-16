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

from __future__ import absolute_import, division, print_function

import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    pairwise_distances, silhouette_score)
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow_similarity.benchmark import Result
from tensorflow_similarity.callbacks.base import MoiraiCallback


class BenchmarkCallback(MoiraiCallback):
    """Generalized Tensorflow Benchmark for Single-Shot Learning experiments.

    Arguments:
        x (dict): A dictionary contains unseen examples that
            we want to benchmark our models against. Will be split into tests
            and targets based on the number of ways and shots specified in Ns
            and ks.
        y (list): A list contains the labels for unseen examples that we want
            to benchmark against. Will be split into tests
            and targets based on the number of ways and shots specified in Ns
            and ks.
        x_key (string): The key into the x dictionary that contains the data
        log_dir (String): The path to the local directory where we
            want to log our benchmark. If None then we will not save to disk.
        file_name (String): Optional, the name of the file we want to
            save the benchmark results to.
        bigquery_path (String): The path to bigquery that we want to save
            the benchmark results to. If None then we will not write to
            bigquery.
        model_information (Dict): The information about the similarity
            model that the users want to record along with the result.
        data_information (Dict): The information about the data that the
            users want to record along with the result.
        Ns (List[int]): The list of number of unseen classes (ways), that
            the users want to record the metrics of. Defaults to [5].
        ks (List[int]): The list of number of targets per unseen class (shots),
            that the users want to record the metrics of. Defaults to [5].
    """

    def __init__(self,
                 x,
                 y,
                 x_key='example',
                 log_dir='/benchmark_metrics',
                 file_name="result.json",
                 bigquery_dataset_id=None,
                 bigquery_table_id=None,
                 model_information=dict(),
                 data_information=dict(),
                 Ns=[5],
                 ks=[5]):

        self.x = x
        self.y = y
        self.log_dir = log_dir
        self.file_name = file_name
        self.bigquery_dataset_id = bigquery_dataset_id
        self.bigquery_table_id = bigquery_table_id

        Ns.sort()
        ks.sort()
        self.Ns = Ns
        self.ks = ks
        self.x_key = x_key

        self.result = Result(model_information, data_information)

        # this is a dictionary of sampled target indices
        # where the keys are label of the target and the values are an array of
        # indices of the medioid targets, when we are doing N-ways,
        # k-shots, we can retrieve the target dataset by query
        # self.sampled_target_indices[label][:k] for each of the label
        # sampled in N.
        self.sampled_target_indices = dict()

        # sampled random classes for each N-ways
        self.sampled_classes = dict()
        unique_classes = np.unique(y)
        targets_sampled = set()
        for N in self.Ns:
            # sampe N unique classes for N ways learning
            sampled_class = np.random.choice(unique_classes, N, replace=False)
            self.sampled_classes[N] = sampled_class

            targets_sampled = targets_sampled.union(set(sampled_class))

        # a dictionary where value is the target label and the value are the
        # list of indices in x/y that has the label
        self.class_indices = dict()

        # iterate through each target sampled and get the top k medoid
        for target_label in targets_sampled:

            sampled_indices = np.where(y == target_label)[0]
            self.class_indices[target_label] = sampled_indices
            sampled_xs = x[sampled_indices]
            num_samples = sampled_xs.shape[0]
            data_shape = sampled_xs.shape[1:]
            flatten_shape = (num_samples, np.product(data_shape))

            flat_sampled_xs = sampled_xs.reshape(flatten_shape)
            distances = pairwise_distances(flat_sampled_xs, flat_sampled_xs)
            medoid_idxs = np.argpartition(
                distances.sum(axis=0), np.arange(self.ks[-1]))
            medoids = [sampled_indices[idx] for idx in medoid_idxs]
            self.sampled_target_indices[target_label] = medoids

        # the key for this dictionary is _make_key(N, k) and the values are
        # the targets indices for N-way k-shots benchmark
        self.targets_indices = dict()
        self.test_indices = dict()

        for N in self.Ns:
            sampled_classes = self.sampled_classes[N]
            for k in self.ks:
                target_indices = []
                test_indices = []

                for label in sampled_classes:
                    all_idx = self.class_indices[label]
                    target_idx = self.sampled_target_indices[label][:k]

                    # the test idx for a label is all the other points except
                    # the ones used for targets
                    test_idx = np.array(list(set(all_idx) - set(target_idx)))
                    target_indices.extend(target_idx)
                    test_indices.extend(test_idx)

                key = (N, k)
                self.targets_indices[key] = target_indices
                self.test_indices[key] = test_indices

    def on_epoch_end(self, epoch, logs={}):
        self.result.training_information["num_epochs"] = epoch + 1
        for N in self.Ns:
            for k in self.ks:
                key = (N, k)
                targets_indices = self.targets_indices[key]
                test_indices = self.test_indices[key]
                y_targets = self.y[targets_indices]
                x_targets = self.x[targets_indices]
                y_test = self.y[test_indices]
                x_test = self.x[test_indices]

                # predict the embeddings of target and test dataset
                inference_time_start = time.time()
                embeddings_targets = self.simhash.predict(
                    {self.x_key: x_targets})
                embeddings_test = self.simhash.predict({self.x_key: x_test})

                # update inference time
                inference_time_end = time.time()
                inference_time = inference_time_end - inference_time_start
                inference_time = round(float(inference_time), 4)
                self.result.update_inference_time(inference_time)

                distances = euclidean_distances(
                    embeddings_test, embeddings_targets)

                y_pred_indices = np.argmin(distances, axis=1)
                y_pred = y_targets[y_pred_indices]

                # compute clustering metrics
                homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
                    y_test, y_pred)
                s_score = silhouette_score(embeddings_test, y_test)

                # compute accuracy
                m = tf.keras.metrics.Accuracy()
                m.update_state(y_test, y_pred)
                accuracy = m.result().numpy()

                # convert those scores into python float as np.float32 is not
                # JSON serializable
                homogeneity = round(float(homogeneity), 4)
                completeness = round(float(completeness), 4)
                v_measure = round(float(v_measure), 4)
                s_score = round(float(s_score), 4)
                accuracy = round(float(accuracy), 4)

                self.result.record_metric("homogeneity", N, k, homogeneity)
                self.result.record_metric("completeness", N, k, completeness)
                self.result.record_metric("v_measure", N, k, v_measure)
                self.result.record_metric("silhouette_score", N, k, s_score)
                self.result.record_metric("accuracy", N, k, accuracy)

                if self.result.should_update_best_metrics(N, k):
                    self.result.update_best_metric("epoch", N, k, epoch + 1)
                    self.result.update_best_metric("accuracy", N, k, accuracy)
                    self.result.update_best_metric(
                        "target_embeddings", N, k, embeddings_targets.tolist())
                    self.result.update_best_metric(
                        "test_embeddings", N, k, embeddings_test.tolist())
                    self.result.update_best_metric(
                        "target_labels", N, k, y_targets.tolist())
                    self.result.update_best_metric(
                        "test_labels", N, k, y_test.tolist())

    def on_train_end(self, logs=None):
        # download to disk if specified local_path
        if self.log_dir:
            self.result.download_to_disk(self.log_dir, self.file_name)
