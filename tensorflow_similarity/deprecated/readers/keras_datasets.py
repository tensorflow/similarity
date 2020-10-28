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
import os
import six
from tensorflow_similarity.utils.config_utils import deserialize_moirai_object, register_custom_object
from tensorflow_similarity.utils.logging_utils import setup_named_logger
import numpy as np
from tensorflow.keras.datasets import *
from tqdm import tqdm


def filter_127(item):
    x, y = item
    return y == 1 or y == 2 or y == 7


def mnist_example_decoder(x, zerotoone=False):
    o = []
    for row in x:
        o_row = []
        for item in row:
            if zerotoone:
                o_row.append([item / 255.0])
            else:
                o_row.append([item])
        o.append(o_row)
    return np.array(o)


def mnist_examples_0_1(x):
    return mnist_example_decoder(x, zerotoone=True)


def mnist_examples_0_255(x):
    return mnist_example_decoder(x, zerotoone=False)


DATASET_READER_FNS = {
    "cifar10": cifar10.load_data,
    "cifar100": cifar100.load_data,
    "imdb": imdb.load_data,
    "reuters": reuters.load_data,
    "mnist": mnist.load_data,
    "mnist_127": mnist.load_data,
    "fashion_mnist": fashion_mnist.load_data,
    "boston_housing": boston_housing.load_data
}

EXAMPLE_DECODER_FNS = {
    "mnist_127": mnist_examples_0_1,
    "mnist": mnist_examples_0_1
}

FILTER_FNS = {
    "mnist_127": filter_127,
}


def singleton_array(y):
    assert (len(y) == 1)
    return y[0]


LABEL_DECODER_FNS = {
    'cifar10': singleton_array,
    'cifar100': singleton_array,
}


class KerasDatasetReader(object):
    def __init__(self, dataset='cifar10', split="train", **kwargs):
        self.dataset = dataset
        self.split = split
        self.custom_args = kwargs

    def subreaders(self):
        return [self]

    def needs_refresh(self):
        return False

    def read(self):
        logger = setup_named_logger("keras_reader")
        reader = DATASET_READER_FNS[self.dataset]

        # ((x_train, y_train), (x_test, y_test))
        data = reader()
        if self.split == 'train':
            shard = 0
        elif self.split == 'test':
            shard = 1
        else:
            raise ValueError('unknown split: %s' % self.split)

        x, y = data[shard]

        if self.dataset in EXAMPLE_DECODER_FNS:
            decoder = EXAMPLE_DECODER_FNS[self.dataset]
            x = [decoder(i) for i in x]

        if self.dataset in LABEL_DECODER_FNS:
            decoder = LABEL_DECODER_FNS[self.dataset]
            y = [decoder(v) for v in y]

        if self.dataset in FILTER_FNS:
            filter = FILTER_FNS[self.dataset]
            x_new = []
            y_new = []

            for x_, y_ in zip(x, y):
                if filter((x_, y_)):
                    x_new.append(x_)
                    y_new.append(y_)

            x = x_new
            y = y_new

        classes_seen = set()

        if self.split == 'test':
            groups = []
            for x_, y_ in zip(x, y):
                if y_ in classes_seen:
                    groups.append("validation_set")
                else:
                    groups.append("targets")
                    classes_seen.add(y_)

            tqdm.write(" %d examples in the test set." % len(x))
            return {'examples': x, 'labels': y, 'groups': groups}
        else:
            g = []
            for _ in x:
                g.append("examples")
            tqdm.write(" %d examples in the train set." % len(x))
            return {'examples': x, 'labels': y, 'groups': g}

    def ready(self):
        return True

    def need_refresh(self):
        return False

    def get_config(self):
        cfg = {
            'class_name': self.__class__.__name__,
            'config': {
                'dataset': self.dataset,
                'split': self.split,
            }
        }
        for k, v in six.iteritems(self.custom_args):
            cfg[k] = v


register_custom_object("KerasDatasetReader", KerasDatasetReader)
