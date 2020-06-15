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

from minipouce import char2seq, char2vec, word2seq, word2vec
from tensorflow_similarity.utils.config_utils import register_custom_object


class MinipoucePreprocessor(object):
    def __init__(self, **kwargs):
        self.config = kwargs

    def get_config(self):
        return {'class_name': self.__class__.__name__, 'config': self.config}


class Identity(MinipoucePreprocessor):
    def __call__(self, x):
        return x


class Char2Seq(MinipoucePreprocessor):
    def __call__(self, x):
        if x is None:
            x = []
        return char2seq(x, **self.config)


class Char2Vec(MinipoucePreprocessor):
    def __call__(self, x):
        if x is None:
            x = []
        return char2vec(x, **self.config)


class Word2Seq(MinipoucePreprocessor):
    def __call__(self, x):
        if x is None:
            x = []
        return word2seq(x, **self.config)


class Word2Vec(MinipoucePreprocessor):
    def __call__(self, x):
        if x is None:
            x = []
        return word2vec(x, **self.config)


register_custom_object("Identity", Identity)
register_custom_object("Char2Seq", Char2Seq)
register_custom_object("Char2Vec", Char2Vec)
register_custom_object("Word2Seq", Word2Seq)
register_custom_object("Word2Vec", Word2Vec)
