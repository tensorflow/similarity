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
from tensorflow_similarity.utils.config_utils import deserialize_moirai_object, register_custom_object
from tensorflow_similarity.utils import logging_utils
import numpy as np
import os


class InitialSample(MoiraiInitializer):
    def __init__(self, sampler=None, existence_reader=None, train_only=True):
        super(InitialSample, self).__init__(train_only=train_only)

        self.sampler = deserialize_moirai_object(sampler)
        self.existence_reader = deserialize_moirai_object(existence_reader)

    def run_initializer(self, callback=None):
        # If either:
        # 1. No reader is specified, or
        # 2. A reader is specified, and indicates it's not ready
        #
        # We perform the initial sample.
        logging_utils.debug("Running initializer.")
        if not self.existence_reader or not self.existence_reader.ready():
            logging_utils.info("Creating initial sample of data.")
            self.sampler(callback=callback)
            logging_utils.info("Done creating initial sample of data.")

    def get_config(self):
        config = super(InitialSample, self).get_config()
        config['config']['sampler'] = self.sampler.get_config()
        config['config'][
            'existence_reader'] = self.existence_reader.get_config(),
        return config


register_custom_object("InitialSample", InitialSample)
