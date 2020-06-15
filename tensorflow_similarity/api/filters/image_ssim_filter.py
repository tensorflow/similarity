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

from tensorflow_similarity.api.engine.filter import MoiraiPairFilter
from tensorflow_similarity.utils.config_utils import register_custom_object
from skimage.measure import compare_ssim


class SSIMDistanceFilter(MoiraiPairFilter):
    def __init__(self,
                 feature_name="examples",
                 min_pos_similarity=0.99,
                 max_neg_similarity=0.98,
                 multichannel=True):
        super(SSIMDistanceFilter, self).__init__()
        self.feature_name = feature_name
        self.min_pos_similarity = min_pos_similarity
        self.max_neg_similarity = max_neg_similarity
        self.multichannel = multichannel

    def _compare(self, a, b):
        if self.feature_name not in a.raw_feature_dictionary:
            print("Unknown feature: %s - valid features are %s" %
                  (self.feature_name, a.value.raw_feature_dictionary.keys()))
        a = a.raw_feature_dictionary[self.feature_name]
        b = b.raw_feature_dictionary[self.feature_name]
        return compare_ssim(a, b, multichannel=self.multichannel)

    def keep_positive(self, a, b):
        return self._compare(a, b) >= self.min_pos_similarity

    def keep_negative(self, a, b):
        return self._compare(a, b) <= self.max_neg_similarity

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'feature_name': self.feature_name,
                'min_pos_similarity': self.min_pos_similarity,
                'max_neg_similarity': self.max_neg_similarity,
                'multichannel': self.multichannel
            }
        }


register_custom_object("SSIMDistanceFilter", SSIMDistanceFilter)
