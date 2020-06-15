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
import glob
import h5py
import tensorflow.keras
from tensorflow.keras.utils import HDF5Matrix
import numpy as np
import os
from tensorflow_similarity.dataset import deserialize_featurespec
from tensorflow_similarity.utils import config_utils
from tensorflow_similarity.readers.base_file_reader import BaseFileReader
from tensorflow_similarity.readers.concatenated_view import ConcatenatedView
from tensorflow_similarity.utils.config_utils import value_or_callable, register_custom_object, deserialize_moirai_object
import six


class H5Reader(BaseFileReader):
    def __init__(self, features=[], **kwargs):
        super(H5Reader, self).__init__(**kwargs)
        self.config = kwargs
        self.features = []
        for f in features:
            self.features.append(deserialize_featurespec(f))

    def read(self):
        files = self.get_files()
        assert len(
            files) > 0, "File pattern not matched: %s" % self.file_pattern

        output = {}
        for feature in self.features:
            shards = []
            for file in files:
                if feature.optional:
                    with h5py.File(file, "r") as t:
                        if feature.name not in t:
                            continue

                shard = HDF5Matrix(
                    file,
                    feature.feature_name,
                    normalizer=feature.preprocessing)
                shards.append(shard)

            if not len(shards):
                continue

            if len(shards) > 1:
                output[feature.name] = ConcatenatedView(shards)
            else:
                output[feature.name] = shards[0]
        return output

    def get_config(self):
        base_config = super(H5Reader, self).get_config()
        base_config["config"]["features"] = []

        for f in self.features:
            base_config["config"]["features"].append(f.get_config())
        return base_config


register_custom_object("H5Reader2", H5Reader)
register_custom_object("H5Reader", H5Reader)


class InMemoryH5Reader(BaseFileReader):
    def __init__(self, **kwargs):
        super(InMemoryH5Reader, self).__init__(**kwargs)
        self.config = kwargs

    def read(self):
        files = self.get_files()
        assert len(
            files) > 0, "File pattern not matched: %s" % self.file_pattern

        output = {}
        for feature in self.features:
            shards = []
            for file in files:
                if feature.optional:
                    with h5py.File(file, "r") as t:
                        if feature.name not in t:
                            continue

                shard = HDF5Matrix(
                    file,
                    feature.feature_name,
                    normalizer=feature.preprocessing)
                shards.append(shard[:])

            if not len(shards):
                continue

            if len(shards) > 1:
                output[feature.name] = ConcatenatedView(shards)
            else:
                output[feature.name] = shards[0]
        return output

    def get_config(self):
        base_config = super(InMemoryH5Reader, self).get_config()
        base_config["config"]["features"] = []

        for f in self.features:
            base_config["config"]["features"].append(f.get_config())
        return base_config


register_custom_object("InMemoryH5Reader2", InMemoryH5Reader)
register_custom_object("InMemoryH5Reader", InMemoryH5Reader)
