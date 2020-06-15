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
from tqdm import tqdm


class TextReader(BaseFileReader):
    def __init__(self,
                 example_feature_name="examples",
                 label_feature_name="labels",
                 group_feature_name="groups",
                 show_progress_bar=False,
                 **kwargs):
        super(TextReader, self).__init__(**kwargs)
        self.example_feature_name = example_feature_name
        self.label_feature_name = label_feature_name
        self.group_feature_name = group_feature_name
        self.show_progress_bar = show_progress_bar

    def read(self):
        files = self.get_files()
        assert len(
            files) > 0, "File pattern not matched: %s" % self.file_pattern

        output = {
            self.example_feature_name: [],
            self.label_feature_name: [],
            self.group_feature_name: []
        }

        for file in files:
            with tf.io.gfile.GFile(file, "r") as f:
                lines = f.read().split("\n")

                for line in tqdm(lines, "Parsing text"):
                    line = line.strip()
                    if len(line) <= 2:
                        continue

                    tokens = line.split(",", 3)

                    group, label, element = tokens[0], tokens[1], tokens[2]

                    output[self.example_feature_name].append(element)
                    output[self.label_feature_name].append(label)
                    output[self.group_feature_name].append(group)

        for k, v in output.items():
            output[k] = np.array(v)
        return output

    def get_config(self):
        base_config = super(TextReader, self).get_config()

        updates = {
            "example_feature_name": self.example_feature_name,
            "label_feature_name": self.label_feature_name,
            "group_feature_name": self.group_feature_name
        }

        base_config["config"].update(updates)
        return base_config


register_custom_object("TextReader", TextReader)
