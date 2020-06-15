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
import multiprocessing
import six
import numpy as np
import numpy.random as rng
import random
from tqdm import tqdm
import traceback
import os
import h5py
import time
import traceback
from tensorflow_similarity.utils.config_utils import value_or_callable, deserialize_moirai_object, register_custom_object
from tensorflow_similarity.utils.preprocess import preprocess
from tensorflow_similarity.utils import logging_utils


class SampleShardTask:
    """Class encapsulating information needed to sample from a single shard."""

    def __init__(self, shard, set_name, reader_config, writer_config,
                 preprocessing_config, augmentation_config, num_examples):
        # Deserialization happens in the process() call, in another process.
        self.shard = shard
        self.set_name = set_name
        self.reader = reader_config
        self.writer = writer_config
        self.preprocessing = preprocessing_config
        self.augmentation = augmentation_config
        self.num_examples = num_examples

    def augment(self, x, y):
        """Apply the configured augmentation to the input data."""
        if self.augmentation is None:
            return x, y

        augmented_xs = []
        augmented_ys = []

        for xe, ye in zip(x, y):
            for a in self.augmentation(xe)[1:]:
                augmented_xs.append(a)
                augmented_ys.append(ye)

        return augmented_xs, augmented_ys

    def prepare_for_write(self, k, v, numeric_sets, string_sets):
        if isinstance(v, np.ndarray):
            if (v.dtype.char == 'S' or v.dtype.char == 'U'
                    or v.dtype.char.startswith("<U")
                    or v.dtype.char.startswith("|U")):
                string_sets[k] = v
            else:
                numeric_sets[k] = v
        elif isinstance(v[0], six.string_types):
            string_sets[k] = v
        else:
            numeric_sets[k] = v

    def process(self):
        self.reader = deserialize_moirai_object(self.reader)
        self.writer = deserialize_moirai_object(self.writer)

        if self.preprocessing:
            self.preprocessing = deserialize_moirai_object(self.preprocessing)
        if self.augmentation:
            self.augmentation = deserialize_moirai_object(self.augmentation)

        x, y = self.reader.read()[self.set_name]
        indices = np.random.choice(len(x), self.num_examples)

        ys = np.take(y, indices, axis=0)
        xs = np.take(x, indices, axis=0)

        aug_xs = []
        aug_ys = []

        if self.augmentation:
            aug_xs, aug_ys = self.augment(xs, ys)

        pp_aug_xs = []
        pp_xs = []
        cnt = 0
        if self.preprocessing:
            for i in aug_xs:
                cnt += 1
                pp_aug_xs.append(self.preprocessing(i))
            for i in xs:
                pp_xs.append(self.preprocessing(i))
                cnt += 1

        string_sets = {}
        numeric_sets = {}

        if self.augmentation:
            if self.preprocessing:
                self.prepare_for_write('x', pp_xs, numeric_sets, string_sets)
                self.prepare_for_write('y', ys, numeric_sets, string_sets)

                self.prepare_for_write('x_aug', pp_aug_xs, numeric_sets,
                                       string_sets)
                self.prepare_for_write('y_aug', aug_ys, numeric_sets,
                                       string_sets)
                self.prepare_for_write('x_aug_raw', aug_xs, numeric_sets,
                                       string_sets)
                self.prepare_for_write('y_aug_raw', aug_ys, numeric_sets,
                                       string_sets)
            else:
                self.prepare_for_write('x_aug', aug_xs, numeric_sets,
                                       string_sets)
                self.prepare_for_write('y_aug', aug_ys, numeric_sets,
                                       string_sets)
        else:
            if self.preprocessing:
                self.prepare_for_write('x', pp_xs, numeric_sets, string_sets)
                self.prepare_for_write('y', ys, numeric_sets, string_sets)
                self.prepare_for_write('x_sample_raw', xs, numeric_sets,
                                       string_sets)
                self.prepare_for_write('y_sample_raw', ys, numeric_sets,
                                       string_sets)
            else:
                self.prepare_for_write('x', xs, numeric_sets, string_sets)
                self.prepare_for_write('y', ys, numeric_sets, string_sets)
        self.writer.get_shard(self.shard).write(numeric_sets, string_sets)
        return xs, ys, aug_xs, aug_ys


def run_task(task):
    try:
        task.process()
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    return ""


def sample(reader,
           set_name,
           writer,
           num_samples=100,
           preprocessing=None,
           augmentation=None,
           callback=None,
           workers=1):
    """Generates a (potentially augmented and preprocessed) sample from the given
    reader, preprocessor, and augmentor.

    Arguments:
        reader - a MoiraiReader responsible for reading the raw data.
        set_name - the data set name to be sampled.
        writer - a MoiraiWriter responsible for writing the sampled data.
        num_samples - the number of examples to sample from
    """
    writer_cfg = writer.get_config()
    subreaders = reader.subreaders()

    tasks = []
    for i, reader in enumerate(subreaders):
        print(reader)
        reader_cfg = reader.get_config()
        tasks.append(
            SampleShardTask(i, set_name, reader_cfg, writer_cfg, preprocessing,
                            augmentation, num_samples))

    if not workers or workers < 2:
        logging.warn("Serial sampling as workers is set to: %s" % workers)
        for task in tasks:
            run_task(task)
        if callback:
            callback([])
    else:
        logging.info("Starting pool with %d workers." % workers)
        with multiprocessing.Pool(workers) as pool:
            if callback:
                pool.map_async(run_task, tasks, 1, callback)
            else:
                for i in tqdm(
                        pool.imap_unordered(run_task, tasks),
                        total=len(tasks),
                        desc="Sampling data."):
                    pass


class Sampler(object):
    def __init__(self,
                 reader=None,
                 set_name=None,
                 writer=None,
                 sample_size=100,
                 preprocessing=None,
                 augmentation=None,
                 workers=None):
        self.reader = deserialize_moirai_object(reader)
        self.dataset_name = value_or_callable(set_name)
        self.writer = deserialize_moirai_object(writer)
        self.sample_size = value_or_callable(sample_size)
        self.preprocessing = deserialize_moirai_object(preprocessing)
        self.augmentation = deserialize_moirai_object(augmentation)
        self.workers = value_or_callable(workers)

    def __call__(self, callback=None, reuse_data=False):
        sample(
            self.reader,
            self.dataset_name,
            self.writer,
            num_samples=self.sample_size,
            callback=callback,
            preprocessing=self.preprocessing,
            augmentation=self.augmentation,
            workers=self.workers)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'reader':
                self.reader.get_config(),
                'writer':
                self.writer.get_config(),
                'set_name':
                self.dataset_name,
                'sample_size':
                self.sample_size,
                'preprocessing':
                self.preprocessing.get_config()
                if self.preprocessing else None,
                'augmentation':
                self.augmentation.get_config() if self.augmentation else None,
                'workers':
                self.workers,
            }
        }


register_custom_object("Sampler", Sampler)
