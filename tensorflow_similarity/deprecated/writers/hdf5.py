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

import h5py
from tensorflow_similarity.utils.logging_utils import get_logger
from tensorflow_similarity.utils.config_utils import value_or_callable, register_custom_object
from tensorflow_similarity.writers.base import ShardedFileWriter
import os


class H5Writer(ShardedFileWriter):
    def get_config(self):
        config = super(H5Writer, self).get_config()
        config['class_name'] = 'H5Writer'
        return config

    def write_file(self, numeric_dataset_dictionary,
                   string_dataset_dictionary):
        tmp_file = "%s.tmp" % self.output_file_pattern
        log = get_logger()
        with h5py.File(tmp_file, "w") as f:
            for name, v in numeric_dataset_dictionary.items():
                try:
                    f.create_dataset(name, data=v)
                except BaseException:
                    log.error("Could not persist dataset: %s", name)
                    log.info("Data was:\n %s" % v)

            str_type = h5py.special_dtype(vlen=str)
            for name, v in string_dataset_dictionary.items():
                f.create_dataset(name, (len(v), ), dtype=str_type)
                f[name][:] = v
        os.rename(tmp_file, self.output_file_pattern)

    def get_config(self):
        config = super(H5Writer, self).get_config()
        config['class_name'] = 'H5Writer'
        return config


register_custom_object("H5Writer", H5Writer)
