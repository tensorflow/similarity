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

from absl import app, flags
import collections
import copy
import logging
import math
import multiprocessing
import os
import sys
import time
import traceback

import h5py
import numpy as np
import six
import sklearn
import tabulate
import tensorflow as tf
from PIL import Image

from termcolor import colored
from tqdm import tqdm

from tensorflow_similarity.api.engine.nearest_neighbors import \
    get_best_available_nearest_neighbor_algorithm
from tensorflow_similarity.callbacks.base import MoiraiCallback
from tensorflow_similarity.dataset import Dataset
from tensorflow_similarity.utils.config_utils import (deserialize_moirai_object,
                                       register_custom_object)
from tensorflow_similarity.utils.io_utils import read_file_pattern, write_hdf5_file
from tensorflow_similarity.utils.logging_utils import get_logger
from tensorflow_similarity.utils.preprocess import preprocess
from tensorflow_similarity.writers.base import Writer
from PIL import Image
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context

FLAGS = flags.FLAGS


def run_task(task):
    return task.process()


def highlight(x, thing, colorize=True):
    configs = {
        "big_gain": ("green", ("reverse", ), "%s ^^" % x),
        "gain": ("cyan", ("reverse", ), "%s ^ " % x),
        "loss": ("yellow", ("reverse", ), "%s v " % x),
        "big_loss": ("red", ("reverse", ), "%s vv" % x),
    }

    if thing not in configs:
        if not colorize:
            return "%s   " % x
        return x

    config = configs[thing]
    if colorize:
        return colored(x, config[0], attrs=[x for x in config[1]])
    else:
        return config[2]


def highlight_metric(v, v_old, lower_is_better=False, colorize=True):
    s = "% 10.4f" % v

    if lower_is_better:
        v, v_old = v_old, v

    if v > v_old * 1.1:
        return highlight(s, "big_gain", colorize=colorize)
    elif v > v_old:
        return highlight(s, "gain", colorize=colorize)
    elif v < v_old * .9:
        return highlight(s, "big_loss", colorize=colorize)
    elif v < v_old:
        return highlight(s, "loss", colorize=colorize)
    else:
        return highlight(s, "neutral", colorize=colorize)


class _ComputeValidationSetMetricsTask(object):
    def __init__(self, name, e_test, y_test, e_targ, y_targ):
        self.name = name
        self.e_test = e_test
        self.y_test = y_test

        self.e_targ = e_targ
        self.y_targ = y_targ

        self.target_map = {}

        for idx, label in enumerate(self.y_targ):
            self.target_map[label] = idx

    def process(self):
        distances = sklearn.metrics.pairwise.euclidean_distances(
            self.e_test, self.e_targ)

        ranks = []
        misses = []

        y_true = []
        y_pred = []

        n_classes = distances.shape[1]
        confusion_matrix = np.zeros((n_classes, n_classes))

        for example_idx, correct_label in enumerate(self.y_test):
            correct_label_idx = self.target_map[correct_label]
            true_distance = distances[example_idx][correct_label_idx]
            rank = 0

            first = True

            best_idx = np.argmin(distances[example_idx])

            y_true.append(correct_label_idx)
            y_pred.append(best_idx)

            for wrong_label_idx, wrong_distance in enumerate(
                    distances[example_idx]):

                confusion_matrix[correct_label_idx][wrong_label_idx] += 1

                if wrong_label_idx == correct_label_idx:
                    continue

                if wrong_distance <= true_distance:

                    if first:
                        misses.append(
                            (self.name, example_idx, correct_label_idx,
                             wrong_label_idx, wrong_distance, true_distance))
                        first = False
                    rank += 1

            ranks.append(rank)

        top1 = 0
        top3 = 0
        top5 = 0
        top10 = 0

        for rank in ranks:
            if rank == 0:
                top1 += 1
            if rank < 3:
                top3 += 1
            if rank < 5:
                top5 += 1
            if rank < 10:
                top10 += 1

        homogeneity, completeness, v_measure = sklearn.metrics.homogeneity_completeness_v_measure(
            y_true, y_pred)

        total = float(len(self.e_test))
        name = self.name
        additional_logs = {}
        additional_logs["%s_top1_acc" % name] = 100 * np.array(top1 / total)
        additional_logs["%s_top3_acc" % name] = 100 * np.array(top3 / total)
        additional_logs["%s_top5_acc" % name] = 100 * np.array(top5 / total)
        additional_logs["%s_top10_acc" % name] = 100 * np.array(top10 / total)
        additional_logs["%s_avg_rank" % name] = np.array(
            float(np.average(ranks)))
        additional_logs["%s_stddev_rank" % name] = np.array(
            float(np.std(ranks)))
        additional_logs["%s_median_rank" % name] = np.array(
            float(np.median(ranks)))
        additional_logs["%s_max_rank" % name] = np.array(float(np.max(ranks)))
        additional_logs["%s_min_rank" % name] = np.array(float(np.min(ranks)))
        additional_logs["%s_homogenity" % name] = np.array(homogeneity)
        additional_logs["%s_completeness" % name] = np.array(completeness)
        additional_logs["%s_v_measure" % name] = np.array(v_measure)

        # TODO figure out a stability metric

        return additional_logs, misses, confusion_matrix


def tabulate_preserving_whitespace(data, header, **kwargs):
    out_header = []
    for item in header:
        if six.PY2:
            if isinstance(item, unicode):
                item = item.encode("utf-8")
        out_header.append(item)

    out_table = []
    for row in data:
        out_row = []
        for item in row:
            if six.PY2:
                if isinstance(item, unicode):
                    item = item.encode("utf-8")
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
            out_row.append(item)
        out_table.append(out_row)

    if hasattr(tabulate, 'PRESERVE_WHITESPACE'):
        prev_tab_val = tabulate.PRESERVE_WHITESPACE
        tabulate.PRESERVE_WHITESPACE = True
        out = tabulate.tabulate(
            out_table, out_header, disable_numparse=True, **kwargs)
        tabulate.PRESERVE_WHITESPACE = prev_tab_val
    else:
        out = tabulate.tabulate(out_table, out_header, **kwargs)

    return out


class ValidationCallback(MoiraiCallback):
    def __init__(self,
                 x_test,
                 y_test,
                 neighborhood_workers=None,
                 output_dir=None,
                 log_dir=None,
                 colorize=True):
        """Create a validation callback for the specified data.

        Arguments:
            x_test {dict} -- validation set name (or "targets") to preprocessed training data suitable for passing to predict().
            y_test {dict} -- validation set name (or "targets") to preprocessed training data suitable for passing to predict() as a target label.

        Keyword Arguments:
            neighborhood_workers {int} -- Number of workers to use to generate the metrics. (default: equal to the number of validation sets)
            output_dir {string} -- Directory in which to write "miss" information. (default: {None})
            colorize {bool} -- Whether to pretty-print the resutlts in a color-coded manner (default: {True})
        """
        super(ValidationCallback, self).__init__()

        print("Validation set:")
        for k, v in six.iteritems(x_test):
            if isinstance(v, dict):
                one_key = list(v.keys())[0]
                l = len(v[one_key])
            elif isinstance(v, np.ndarray):
                l = v.shape[0]
            else:
                l = len(v)

            print("  %40s: %d" % (k, l))

        self.neighborhood_workers = neighborhood_workers
        if not neighborhood_workers:
            self.neighborhood_workers = len(x_test)

        self.colorize = colorize
        self.nn_cls = get_best_available_nearest_neighbor_algorithm()
        self.history = collections.defaultdict(list)
        self.last_logs = None
        self.output_dir = output_dir
        self.tensorboard_writer = None

        if log_dir:
            self.tensorboard_writer = summary_ops_v2.create_file_writer_v2(
                log_dir)

        self.x_test = x_test
        self.y_test = y_test

        self.x_targets = x_test["targets"]
        self.y_targets = y_test["targets"]
        del self.x_test['targets']
        del self.y_test['targets']

        self.validation_set_names = [x for x in self.x_test.keys()]

    def on_train_end(self, logs={}):
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

    def on_epoch_end(self, epoch, logs={}):
        e_targets = self.simhash.predict(self.x_targets)
        additional_logs = collections.defaultdict(float)
        tasks = []
        all_misses = []

        groups = sorted(self.x_test.keys())
        for k in groups:
            x_test = self.x_test[k]
            y_test = self.y_test[k]

            e_test = self.simhash.predict(x_test)
            tasks.append(
                _ComputeValidationSetMetricsTask(k, e_test, y_test, e_targets,
                                                 self.y_targets))

        if self.neighborhood_workers < 2:
            aggregated_confusion = None
            for task in tasks:
                d, misses, confusion = run_task(task)
                if aggregated_confusion is None:
                    aggregated_confusion = np.zeros_like(confusion)
                aggregated_confusion += confusion
                all_misses.extend(misses)
                for k, v in d.items():
                    additional_logs[k] = v
        else:
            pool = multiprocessing.Pool(self.neighborhood_workers)
            aggregated_confusion = None
            for d, misses, confusion in pool.imap_unordered(run_task, tasks):
                if aggregated_confusion is None:
                    aggregated_confusion = np.zeros_like(confusion)
                aggregated_confusion += confusion

                all_misses.extend(misses)
                for k, v in d.items():
                    additional_logs[k] = v
            pool.close()

        if self.tensorboard_writer:
            self.visualize_confusion(aggregated_confusion)

        table_headers = [
            'Dataset', 'V-Measure', 'Top1%', 'Top5%', 'Top10%', 'AvgR',
            'MedianR', 'MaxR', 'MinR'
        ]

        table_data = []
        for name in sorted([x for x in self.validation_set_names]):
            output = [name]

            for key in ["v_measure", "top1_acc", "top5_acc", "top10_acc"]:
                v = additional_logs["%s_%s" % (name, key)]
                if self.last_logs:
                    v_old = self.last_logs["%s_%s" % (name, key)]
                    s = highlight_metric(v, v_old, colorize=self.colorize)
                else:
                    s = highlight_metric(v, v, colorize=self.colorize)

                output.append(s)

            for key in ["avg_rank", "median_rank", "max_rank", "min_rank"]:
                v = additional_logs["%s_%s" % (name, key)]
                if self.last_logs:
                    v_old = self.last_logs["%s_%s" % (name, key)]
                    s = highlight_metric(
                        v, v_old, lower_is_better=True, colorize=self.colorize)
                else:
                    s = highlight_metric(
                        v, v, lower_is_better=True, colorize=self.colorize)
                output.append(s)

            table_data.append(output)

        tqdm.write("\n")

        tqdm.write(
            tabulate_preserving_whitespace(
                table_data,
                table_headers,
                tablefmt=FLAGS.tabulate_table_format))

        tqdm.write("[%s] [%s] [%s] [%s]" % (
            highlight("Major Improvement", "big_gain", colorize=self.colorize),
            highlight("Improvement", "gain", colorize=self.colorize),
            highlight("Degradation", "loss", colorize=self.colorize),
            highlight("Major Degradation", "big_loss",
                      colorize=self.colorize)))

        if self.output_dir:
            with tf.io.gfile.GFile(
                    os.path.join(self.output_dir, "epoch_%d_misses" % epoch),
                    "w") as f:
                header = [
                    "Group", "Example", "Correct Label", "Mispredicted Label",
                    "Correct Distance", "Misprediction Distance"
                ]
                table = []

                for (name, example_idx, right_label_idx, wrong_label_idx,
                     wrong_distance, true_distance) in all_misses:
                    # Translate from indxes-within-group to index-within-dataset
                    ex_id = self.ids_by_group[name][example_idx]
                    right_id = self.ids_by_group["targets"][right_label_idx]
                    wrong_id = self.ids_by_group["targets"][wrong_label_idx]
                    table.append([
                        name, ex_id, right_id, wrong_id, true_distance,
                        wrong_distance
                    ])

                f.write(
                    tabulate_preserving_whitespace(
                        table, header, tablefmt=FLAGS.tabulate_table_format))

        for k, v in additional_logs.items():
            logs[k] = v

        self.last_logs = additional_logs

    def visualize_confusion(self, confusion):

        confusion = np.array(confusion)
        confusion_shape = confusion.shape

        final_img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(final_img)

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                with self.tensorboard_writer.as_default():
                    summary_ops_v2.image(name="confusion_map",
                                         tensor=tf.convert_to_tensor(
                                             final_img),
                                         family="visualizations", step=batch_id)

    def get_config(self):
        x_test = {}
        for k, v in six.iteritems(self.x_test()):
            x_test[k] = v.to_list()

        y_test = {}
        for k, v in six.iteritems(self.y_test()):
            y_test[k] = v.to_list()

        return {
            'class_name': 'ValidationCallback',
            'config': {
                'x_test': x_test,
                'y_test': y_test,
                'neighborhood_workers': self.neighborhood_workers,
                'output_dir': self.output_dir,
                'colorize': self.colorize
            }
        }
