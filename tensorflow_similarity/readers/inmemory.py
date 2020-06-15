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
import numpy as np
import os
from tensorflow_similarity.utils import config_utils
from tensorflow_similarity.readers.base_file_reader import BaseFileReader
from tensorflow_similarity.utils.config_utils import register_custom_object, value_or_callable, deserialize_moirai_object
import tensorflow as tf


class MemoryReader(BaseFileReader):
    def __init__(self,
                 examples=[],
                 labels=[],
                 groups=[],
                 example_feature_name="examples",
                 label_feature_name="labels",
                 group_feature_name="groups",
                 **kwargs):
        super(MemoryReader, self).__init__(**kwargs)

        if not isinstance(examples, np.ndarray):
            examples = np.asarray(examples)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        if not isinstance(examples, np.ndarray):
            groups = np.asarray(groups)

        self.examples = examples
        self.labels = labels
        self.groups = groups

        self.example_feature_name = "examples"
        self.label_feature_name = label_feature_name
        self.group_feature_name = group_feature_name

    def read(self):
        output = {
            self.example_feature_name: self.examples,
            self.label_feature_name: self.labels,
            self.group_feature_name: self.groups
        }

        return output

    def get_config(self):
        base_config = super(MemoryReader, self).get_config()
        base_config["config"][
            "example_feature_name"] = self.example_feature_name
        base_config["config"]["label_feature_name"] = self.label_feature_name
        base_config["config"]["group_feature_name"] = self.group_feature_name

        base_config["config"]["examples"] = self.examples.to_list()
        base_config["config"]["labels"] = self.labels.to_list()
        base_config["config"]["groups"] = self.groups.to_list()
        return base_config


register_custom_object("MemoryReader", MemoryReader)
