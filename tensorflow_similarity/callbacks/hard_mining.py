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
import cProfile
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from threading import Lock, Thread

import h5py
import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import model_from_json
from termcolor import colored
from tqdm import tqdm
import tempfile
from tensorflow_similarity.api.engine.nearest_neighbors import \
    get_best_available_nearest_neighbor_algorithm
from tensorflow_similarity.callbacks.base import MoiraiCallback
from tensorflow_similarity.dataset import Dataset, Datum
from tensorflow_similarity.utils.config_utils import (
    deserialize_moirai_object, register_custom_object)
from tensorflow_similarity.utils.io_utils import read_file_pattern, write_hdf5_file
from tensorflow_similarity.utils.logging_utils import get_logger
from tensorflow_similarity.utils.preprocess import preprocess
from tensorflow_similarity.writers.base import Writer

FLAGS = flags.FLAGS

WRITE_LOCK = Lock()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AsyncWriter(Thread):
    def __init__(self, callback):
        super(AsyncWriter, self).__init__()
        self.callback = callback

    def run(self):
        global WRITE_LOCK
        try:
            tmp_file = os.path.join(
                str(self.callback.hard_mining_directory), "tmp_blackboard.data")
            final_file = os.path.join(
                str(self.callback.hard_mining_directory), "blackboard.data")

            output_dict = {"idx": {}, "labels": {}, "embedding": {}}

            WRITE_LOCK.acquire()
            tower_indexes = self.callback.tower_indexes
            tower_embeddings = self.callback.tower_embeddings
            tower_labels = self.callback.tower_embeddings

            self.callback.clear_queues()
            WRITE_LOCK.release()

            for tower, value in six.iteritems(self.callback.tower_indexes):
                output_dict["idx"][tower] = list(copy.deepcopy(value))
            for tower, value in six.iteritems(self.callback.tower_embeddings):
                output_dict["embedding"][tower] = list(copy.deepcopy(value))
            for tower, value in six.iteritems(self.callback.tower_labels):
                output_dict["labels"][tower] = list(copy.deepcopy(value))

            with open(tmp_file, "w") as f:
                json.dump(output_dict, f, cls=NumpyEncoder)
            if os.path.exists(final_file):
                os.remove(final_file)
            os.rename(tmp_file, final_file)

        except BaseException:
            traceback.print_exc()
        finally:
            self.callback.writer = None


class ResultWriter(MoiraiCallback):
    def __init__(self,
                 towers=[],
                 hard_mining_directory=None,
                 write_embeddings=True,
                 queue_size=10000,
                 dump_losses=False):
        self.hard_mining_directory = hard_mining_directory
        if not hard_mining_directory:
            self.hard_mining_directory = tempfile.mkdtemp()
        self.towers = sorted(towers)
        self.tuples_produced = 0
        self.tower_indexes = {}
        self.tower_embeddings = {}
        self.tower_labels = {}
        self.epoch = 0
        self.writer = None
        self.hard_tuple_queue_size = queue_size
        self.write_embeddings = write_embeddings
        self.dump_losses = dump_losses

        for tower in towers:
            self.tower_indexes[tower] = collections.deque(
                [], self.hard_tuple_queue_size)
            self.tower_embeddings[tower] = collections.deque(
                [], self.hard_tuple_queue_size)
            self.tower_labels[tower] = collections.deque(
                [], self.hard_tuple_queue_size)

    def clear_queues(self):
        self.tower_indexes = {}
        self.tower_embeddings = {}
        self.tower_labels = {}
        for tower in self.towers:
            self.tower_indexes[tower] = collections.deque(
                [], self.hard_tuple_queue_size)
            self.tower_embeddings[tower] = collections.deque(
                [], self.hard_tuple_queue_size)
            self.tower_labels[tower] = collections.deque(
                [], self.hard_tuple_queue_size)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'towers': self.towers,
                'write_embeddings': self.write_embeddings,
                'queue_size': self.hard_tuple_queue_size,
                'dump_losses': self.dump_losses
            }
        }

    def consume_pseudometrics(self, logs):
        clear_set = set()
        clear_set.add("embedding_loss")

        tower_outputs = {}
        for tower in self.towers + ["meta"]:
            tower_outputs[tower] = {}

        tower_match = False
        for k, v in six.iteritems(logs):
            if k.startswith("_moirai_pseudometric_"):
                clear_set.add(k)
                k = k[len("_moirai_pseudometric_"):]
                for t in self.towers + ["meta"]:
                    if k.startswith("%s_" % t):
                        tower_match = True
                        r = k.split("_", 1)
                        if len(r) == 2:
                            tower, feature = r
                            tower_outputs[tower][feature] = v
                        else:
                            tower_outputs[None][k] = v
        for c in clear_set:
            if c in logs:
                del logs[c]
        return tower_outputs

    def maybe_start_writer(self):
        if self.writer is None:
            self.writer = AsyncWriter(self)
            self.writer.start()
            self.tuples_produced = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def _is_hard_mining(self, results):
        if "meta" in results and "is_hard" in results["meta"]:
            return True
        return False

    def on_epoch_end(self, epoch, logs={}):
        results = self.consume_pseudometrics(logs)
        self.maybe_start_writer()
        hard_indices = []

        if self._is_hard_mining(results):
            is_hard = results["meta"]["is_hard"]

            for idx, h in enumerate(is_hard):
                if h[0]:
                    hard_indices.append(idx)
            logs["inversion_fraction"] = np.array(
                [len(hard_indices) / float(len(is_hard))])

    def on_batch_end(self, batch, logs={}):
        results = self.consume_pseudometrics(logs)

        if self.dump_losses:
            header = ("Name", "Value")
            data = []
            data.append(("Label(Pos)", results["pos"]["label"][0]))

            embs = []
            for tower in self.towers:
                embs.append((tower, results["pos"]["tower_embedding"][0]))

            data.append(("E(Pos)", str(embs)))

            data.append(("E(Pos)", results["pos"]))

            data.append(("E(Anchor)", results["anchor"]["tower_embedding"][0]))
            data.append(("E(Neg)", results["neg"]["tower_embedding"][0]))
            data.append(("E(Neg2)", results["neg2"]["tower_embedding"][0]))
            data.append(
                ("D(Pos, Anchor)", results["meta"]["pos_anchor_distance"][0]))
            data.append(
                ("D(Pos, Neg)", results["meta"]["pos_neg_distance"][0]))
            data.append(
                ("D(Neg, Neg2)", results["meta"]["neg_neg2_distance"][0]))
            data.append(
                ("Triplet Loss", results["meta"]["triplet_loss_many"][0]))
            data.append(
                ("Quadruplet Loss", results["meta"]["quad_loss_many"][0]))
            import tabulate
            tqdm.write(tabulate.tabulate(data, header, tablefmt="grid"))

        if self._is_hard_mining(results):
            hard_indices = []
            is_hard = results["meta"]["is_hard"]

            for idx, h in enumerate(is_hard):
                if h[0]:
                    hard_indices.append(idx)
            logs["inversion_fraction"] = np.array(
                [len(hard_indices) / float(len(is_hard))])

            for tower in self.towers:
                if "idx" not in results[tower]:
                    continue

                tower_indices = [int(x[0]) for x in results[tower]["idx"]]
                embeddings = [x for x in results[tower]["tower_embedding"]]
                labels = [int(x[0]) for x in results[tower]["label"]]

                global WRITE_LOCK
                WRITE_LOCK.acquire()

                for idx, (ti, e, l) in enumerate(
                        zip(tower_indices, embeddings, labels)):
                    if idx in hard_indices:
                        self.tower_indexes[tower].append(ti)
                        if self.write_embeddings:
                            self.tower_embeddings[tower].append(e)
                        self.tower_labels[tower].append(l)
                WRITE_LOCK.release()

            self.tuples_produced += len(hard_indices)

            if self.tuples_produced > self.hard_tuple_queue_size:
                self.maybe_start_writer()
