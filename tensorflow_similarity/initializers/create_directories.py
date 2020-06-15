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

from tensorflow_similarity.initializers.base import MoiraiInitializer
from tensorflow_similarity.utils.config_utils import value_or_callable, register_custom_object
import tensorflow as tf


class CreateDirectories(MoiraiInitializer):
    def __init__(self, directories=[], train_only=True):
        super(CreateDirectories, self).__init__(train_only=train_only)
        self.directories = directories

    def run_initializer(self):
        for directory in self.directories:
            if not tf.io.gfile.Exists(directory):
                tf.io.gfile.MakeDirs(directory)

    def get_config(self):
        config = super(CreateDirectories, self).get_config()
        config['config']['directories'] = self.directories,
        return config


register_custom_object("CreateDirectories", CreateDirectories)
