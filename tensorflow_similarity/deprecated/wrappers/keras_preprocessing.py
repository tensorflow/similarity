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
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow_similarity.utils.config_utils import register_custom_object
import numpy as np


def pad_sequence(seq, target_length):
    length = len(seq)
    if length < target_length:
        seq = seq + [' '] * (target_length - length)
    elif length > target_length:
        seq = seq[-target_length:]
    return seq


class TextToWordSequence(object):
    def __init__(self, length, **kwargs):
        self.config = kwargs
        self.length = length

    def get_config(self):
        cfg = copy.copy(cfg)
        cfg['length'] = self.length

        return {'class_name': self.__class__.__name__, 'config': cfg}

    def __call__(self, x):
        seq = text_to_word_sequence(x, **self.config)
        np.array(pad_sequence(seq, self.length))


register_custom_object('TextToWordSequence', TextToWordSequence)


class TextToCharSequence(object):
    def __init__(self, length):
        self.length = length

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'length': self.length
            }
        }

    def __call__(self, x):
        seq = list(x)
        out = pad_sequence(seq, self.length)
        return np.array(out)


register_custom_object('TextToCharSequence', TextToCharSequence)
