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
from tensorflow_similarity.utils.config_utils import deserialize_moirai_object
from tensorflow_similarity.dataset import deserialize_featurespec


class Reader(object):
    def __init__(self, features=[]):
        self.features = []
        for f in features:
            self.features.append(deserialize_featurespec(f))

    def subreaders(self):
        return [self]

    def read(self):
        """Returns a dictionary of dataset name to (x, y) datasets."""
        pass

    def ready(self):
        """Returns true iff data is available to read."""
        pass

    def need_refresh(self):
        return False

    def get_config(self):
        feature_config = []
        for f in self.features:
            feature_config.append(f.get_config())

        return {
            'class_name': self.__class__.__name__,
            'config': {
                'features': [feature_config]
            }
        }
