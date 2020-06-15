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

import collections
import copy
import json
import os
import sys
import traceback
import shutil
from threading import Lock, Thread
from tqdm import tqdm
import numpy as np
import six

from tensorflow_similarity.api.engine.callbacks import MoiraiCallback


class RenameMetric(MoiraiCallback):
    def __init__(self, original_name, new_name):
        self.original_name = original_name
        self.new_name = new_name

    def rename(self, logs):
        if self.original_name in logs:
            v = logs[self.original_name]
            del logs[self.original_name]
            logs[self.new_name] = v

    def on_batch_end(self, batch, logs={}):
        self.rename(logs)
        v = logs[self.new_name]

    def on_epoch_end(self, epoch, logs={}):
        self.rename(logs)


class RemoveSuppressedMetrics(MoiraiCallback):
    def __init__(self, suppressed_metric_prefixes=[]):
        self.suppressed_metric_prefixes = suppressed_metric_prefixes

    def __repr__(self):
        metrics = ", ".join(self.suppressed_metric_prefixes)
        return "RemoveSuppressedMetrics([%s])" % metrics

    def _clear(self, logs):
        clear_set = set()

        for k, v in logs.items():
            for prefix in self.suppressed_metric_prefixes:
                if k.startswith(prefix):
                    clear_set.add(k)

        for name in clear_set:
            del logs[name]

    def on_batch_end(self, batch, logs={}):
        self._clear(logs)

    def on_epoch_end(self, batch, logs={}):
        self._clear(logs)


WRITE_LOCK = Lock()


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class _AsyncWriter(Thread):
    def __init__(self, callback, output_dir):
        super(_AsyncWriter, self).__init__()
        self.output_dir = output_dir
        self.callback = callback

    def run(self):
        global WRITE_LOCK
        try:
            tmp_file = os.path.join(self.output_dir, "tmp_blackboard.data")
            final_file = os.path.join(self.output_dir, "blackboard.data")

            output_dict = {"idx": {}, "labels": {}, "embedding": {}}

            WRITE_LOCK.acquire()
            tower_indexes = self.callback.tower_indexes
            self.callback.clear_queues()
            WRITE_LOCK.release()

            for tower, value in six.iteritems(tower_indexes):
                output_dict["idx"][tower] = list(copy.deepcopy(value))

            with open(tmp_file, "w") as f:
                json.dump(output_dict, f, cls=_NumpyEncoder)

            shutil.move(tmp_file, final_file)

        except BaseException:
            traceback.print_exc()
        finally:
            self.callback.writer = None


class ResultWriter(MoiraiCallback):
    def __init__(self,
                 towers=[],
                 hard_mining_directory=None,
                 queue_size=10000):
        self.towers = sorted(towers)
        self.tuples_produced = 0
        self.tower_indexes = {}
        self.writer = None
        self.hard_tuple_queue_size = queue_size
        self.hard_mining_directory = hard_mining_directory
        self.clear_queues()

    def clear_queues(self):
        self.tower_indexes = {}
        self.tower_embeddings = {}
        self.tower_labels = {}
        for tower in self.towers:
            self.tower_indexes[tower] = collections.deque(
                [], self.hard_tuple_queue_size)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'towers': self.towers,
                'queue_size': self.hard_tuple_queue_size,
            }
        }

    def parse_pseudometrics(self, logs):

        tower_outputs = {}
        for tower in self.towers + ["meta"]:
            tower_outputs[tower] = {}

        for k, v in six.iteritems(logs):
            tokens = k.split("_")
            if tokens[-1] != "pseudometric":
                continue

            if len(tokens) < 2:
                continue

            tower = tokens[0]

            if tower == "meta" or tower in self.towers:
                # Drop the tower name at the front, and the _pseudometric at the
                # end.
                feature = "_".join(tokens[1:-1])
                tower_outputs[tower][feature] = v

        return tower_outputs

    def maybe_start_writer(self):
        global WRITE_LOCK

        WRITE_LOCK.acquire()
        if self.writer is None:
            self.writer = _AsyncWriter(
                self, output_dir=self.hard_mining_directory)
            self.writer.start()
            self.tuples_produced = 0
        WRITE_LOCK.release()

    def _is_hard_mining(self, results):
        if "meta" not in results:
            return False

        if "is_hard" in results["meta"]:
            return True

        return False

    def on_epoch_end(self, epoch, logs={}):
        results = self.parse_pseudometrics(logs)
        self.maybe_start_writer()
        hard_indices = []

        if self._is_hard_mining(results):
            is_hard = results["meta"]["is_hard"]

            for idx, h in enumerate(is_hard):
                if h[0]:
                    hard_indices.append(idx)
            logs["inversion_fraction"] = np.float(
                len(hard_indices) / float(len(is_hard)))

    def on_batch_end(self, batch, logs={}):
        results = self.parse_pseudometrics(logs)

        if self._is_hard_mining(results):
            is_hard = results["meta"]["is_hard"]
            num_hard = np.sum(is_hard)
            logs["inversion_fraction"] = np.float(
                num_hard / float(len(is_hard)))

            for tower in self.towers:
                if "idx_out" not in results[tower]:
                    continue

                # Locked section - pull the proper indices.
                global WRITE_LOCK
                WRITE_LOCK.acquire()
                tower_indices = results[tower]["idx_out"]
                for tower_index, hard in zip(tower_indices, is_hard):
                    if hard[0]:
                        self.tower_indexes[tower].append(int(tower_index[0]))
                WRITE_LOCK.release()

            self.tuples_produced += num_hard

            if self.tuples_produced > self.hard_tuple_queue_size / 2.0:
                self.maybe_start_writer()
