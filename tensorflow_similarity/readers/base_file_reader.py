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

import copy
import h5py
import numpy as np
import os
from tensorflow_similarity.utils import config_utils
from tensorflow_similarity.readers.base import Reader
from tensorflow_similarity.utils.config_utils import value_or_callable, deserialize_moirai_object
import tensorflow as tf


class BaseFileReader(Reader):
    def __init__(self,
                 features=[],
                 file_pattern=None,
                 preprocessing=None,
                 sentinel=None,
                 **kwargs):
        super(BaseFileReader, self).__init__(features=features)

        self.config = kwargs
        self.file_pattern = value_or_callable(file_pattern)
        self.sentinel_filename = value_or_callable(sentinel)
        if sentinel:
            self.last_refresh = self.check_last_update(sentinel)

        self.preprocessing = preprocessing
        if preprocessing:
            self.preprocessing = deserialize_moirai_object(preprocessing)

    def check_last_update(self):
        if not tf.io.gfile.Exists(self.sentinel_filename):
            return -1
        return tf.io.gfile.Stat(filename)[8]

    def ready(self):
        print(self.file_pattern)
        return len(tf.io.gfile.glob(self.file_pattern)) > 0

    def get_config(self):
        config = self.config.copy()
        config['file_pattern'] = self.file_pattern
        if self.preprocessing:
            config['preprocessing'] = self.preprocessing.get_config()
        config['sentinel'] = self.sentinel_filename
        return {'class_name': self.__class__.__name__, 'config': config}

    def clone_for_file(self, file):
        file_reader = copy.copy(self)
        file_reader.file_pattern = file
        return file_reader

    def subreaders(self):
        output = []
        files = self.get_files()

        for file in files:
            output.append(self.clone_for_file(file))
        return output

    def get_file(self):
        f = self.get_files()

        assert len(
            f) == 1, "Calling get_file() on a reader with multiple files."
        return f[0]

    def get_files(self):
        assert self.file_pattern is not None
        return tf.io.gfile.glob(self.file_pattern)

    def needs_refresh(self):
        if not self.sentinel_filename:
            return False
        t = self.check_last_update()
        if t > last_refresh:
            self.last_refresh = t
            return True
        return False

    def get_features(self):
        if 'sets' not in self.config:
            return [("examples", ['x', 'y', 'metadata'])]
        return self.config['sets']

    def read(self):
        pass
