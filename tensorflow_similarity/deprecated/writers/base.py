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
from tensorflow_similarity.utils.config_utils import deserialize_moirai_object
import numpy as np


class Writer(object):
    def __init__(self):
        pass

    def write(self, numeric_dataset_dictionary, string_dataset_dictionary):
        raise NotImplementedError()


class ShardedFileWriter(Writer):
    def __init__(self,
                 output_file_pattern=None,
                 num_shards=1,
                 shard_id=None,
                 **kwargs):
        self.kwargs = kwargs
        self.output_file_pattern = deserialize_moirai_object(
            output_file_pattern)
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.shard_filenames = None

    def get_config(self):
        config = self.kwargs.copy()
        config['output_file_pattern'] = self.output_file_pattern
        config['num_shards'] = self.num_shards
        config['shard_id'] = self.shard_id

        return {
            'config': config,
        }
        return config

    def num_shards(self):
        return num_shards

    def get_shard(self, shard_id):
        writer = copy.copy(self)
        writer.num_shards = 1
        writer.shard_id = 0
        writer.shard_filenames = None

        writer.output_file_pattern = self.output_file_pattern.replace(
            "*", str(shard_id))
        return writer

    def write(self, numeric_dataset_dictionary, string_dataset_dictionary):
        if self.num_shards > 1:
            split_numerics = []
            split_strings = []
            for i in range(self.num_shards):
                split_numerics.append({})
                split_strings.append({})

            for i, (key,
                    value) in enumerate(numeric_dataset_dictionary.items()):
                split_numerics[i][key] = np.array_split(value)
            for i, (key,
                    value) in enumerate(string_dataset_dictionary.items()):
                split_strings[i][key] = np.array_split(value)

            for i in range(self.num_shards):
                self.get_shard(i).write_file(split_numerics[i],
                                             split_strings[i])
        else:
            self.write_file(numeric_dataset_dictionary,
                            string_dataset_dictionary)

    def write_file(self, filename, numeric_dataset_dictionary,
                   string_dataset_dictionary):
        raise NotImplementedError()
