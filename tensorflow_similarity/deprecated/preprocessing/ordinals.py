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

import numpy as np
from tensorflow_similarity.preprocessing.base import MoiraiPreprocessor
from tensorflow_similarity.utils.config_utils import register_custom_object


class TextToOrdinals(MoiraiPreprocessor):
    def __init__(self, maxlen=32):
        super(TextToOrdinals, self).__init__(maxlen=maxlen)
        self.maxlen = maxlen

    def __call__(self, text):
        tokens = list(str(text))
        while len(tokens) < self.maxlen:
            tokens.append(' ')
        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]

        return np.array([ord(char) for char in tokens])


register_custom_object("TextToOrdinals", TextToOrdinals)
