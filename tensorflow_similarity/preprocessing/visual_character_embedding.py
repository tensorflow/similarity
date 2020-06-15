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

from absl import app, flags
import base64
import io
import json
import math
import numpy as np
import os
import random
import six
from tensorflow_similarity.utils.config_utils import register_custom_object
import tensorflow as tf
import zlib

flags.DEFINE_string("visual_character_embedding_dir",
                    "%s/data/" % os.path.dirname(__file__),
                    "Directory in which visual embedding data is stored.")

FLAGS = flags.FLAGS


def encode_img(img):
    arr = np.array(img)
    f = io.BytesIO()
    np.save(f, arr)
    compressed = base64.b64encode(zlib.compress(f.getvalue())).decode("ascii")
    return compressed


def decode_img(value):
    value = base64.b64decode(value)
    value = zlib.decompress(value)
    f = io.BytesIO(value)
    return np.load(f)


def stitch_imgs(img_list, array_dims, character_dims, channels=3):
    output = np.zeros((array_dims[0] * character_dims[0],
                       array_dims[1] * character_dims[1], channels))
    for idx, img in enumerate(img_list):
        x_idx = idx % array_dims[0]
        y_idx = int(idx / array_dims[0])
        assert y_idx < array_dims[1]

        x_off = x_idx * character_dims[0]
        y_off = y_idx * character_dims[1]

        x_limit = x_off + character_dims[0]
        y_limit = y_off + character_dims[1]

        output[x_off:x_off + character_dims[0], y_off:y_off +
               character_dims[1], :] = img

    return output


class VisualCharacterEmbedding(object):
    def __init__(self, dim=32, maxlen=32):

        self.maxlen = 32
        self.dim = dim
        self.fulldim = [self.maxlen]

        if isinstance(self.dim, int):
            self.fulldim.append(self.dim)
            self.file_suffix = "%s" % self.dim
        else:
            strs = []
            for d in self.dim:
                self.fulldim.append(d)
                strs.append("%s" % d)
            self.file_suffix = "_".join(strs)

        self.fulldim = tuple(self.fulldim)

        embedding_filename = "%s/vis_embed_trained_%s" % (
            FLAGS.visual_character_embedding_dir, self.file_suffix)

        with tf.io.gfile.GFile(embedding_filename, "r") as f:
            self.char_to_embedding = json.load(f)
        for k, v in six.iteritems(self.char_to_embedding):
            self.char_to_embedding[k] = np.array(v)

    def __call__(self, x):
        if x is None:
            return np.zeros(self.fulldim)

        chars = len(x)
        output = []
        if len(x) > self.maxlen:
            x = x[:self.maxlen]

        for i in x:
            output.append(self.char_to_embedding.get(i, np.zeros(self.dim)))
        while len(output) < self.maxlen:
            output.append(np.zeros(self.dim))
        return np.array(output)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'maxlen': self.maxlen,
                'dim': self.dim
            }
        }


class VisualCharacterEmbedding32(VisualCharacterEmbedding):
    def __init__(self, dim=None, maxlen=32):
        super(VisualCharacterEmbedding32, self).__init__(dim=32, maxlen=maxlen)


register_custom_object("VisualCharacterEmbedding32",
                       VisualCharacterEmbedding32)


class VisualCharacterEmbedding16(VisualCharacterEmbedding):
    def __init__(self, maxlen=32):
        super(VisualCharacterEmbedding16, self).__init__(dim=16, maxlen=maxlen)


register_custom_object("VisualCharacterEmbedding16",
                       VisualCharacterEmbedding16)


class VisualCharacterEmbedding4(VisualCharacterEmbedding):
    def __init__(self, maxlen=32):
        super(VisualCharacterEmbedding4, self).__init__(dim=4, maxlen=maxlen)


register_custom_object("VisualCharacterEmbedding4", VisualCharacterEmbedding4)


class VisualCharacterEmbedding64x64(VisualCharacterEmbedding):
    def __init__(self, maxlen=32):
        super(VisualCharacterEmbedding64x64, self).__init__(
            dim=(64, 64), maxlen=maxlen)


register_custom_object("VisualCharacterEmbedding64x64",
                       VisualCharacterEmbedding64x64)


class VisualCharacters(object):
    def __init__(self,
                 maxlen=32,
                 embedding_filename=None,
                 zero_to_one=False,
                 inflate=True):

        self.maxlen = maxlen
        self.inflate = inflate
        self.zero_to_one = zero_to_one
        if not embedding_filename:
            self.embedding_filename = "%s/visual_characters" % FLAGS.visual_character_embedding_dir
        else:
            self.embedding_filename = "%s/%s" % (
                FLAGS.visual_character_embedding_dir, embedding_filename)

        with tf.io.gfile.GFile(self.embedding_filename, "r") as f:
            data = json.load(f)
            self.character_map = data['character_map']
            self.font_map = data['font_map']
            self.embeddings = decode_img(data['embeddings'])

            if self.zero_to_one:
                self.embeddings = self.embeddings / 255.0

            self.num_choices = len(self.font_map)
            # Generates the output image as if it were a string rendered left
            # to right.
            self.array_dims = (self.maxlen, 1)
            self.dim = np.shape(self.embeddings[0][0])

        if self.inflate:
            self.channels = 3
        else:
            self.channels = 1

        self.output_dim = tuple([
            self.array_dims[0] * self.dim[0], self.array_dims[1] * self.dim[1],
            self.channels
        ])
        self.fulldim = tuple([self.dim[0], self.dim[1], self.channels])

    def get_embedding(self, char):
        if char not in self.character_map:
            return np.zeros([self.dim[0], self.dim[1], self.channels])

        idx = self.character_map[char]

        val = random.choice(self.embeddings[idx])

        output = np.zeros(self.fulldim)
        for x, xv in enumerate(val):
            for y, yv in enumerate(xv):
                if self.inflate:
                    output[x][y] = [yv, yv, yv]
                else:
                    output[x][y] = [yv]

        return output

    def __call__(self, x):
        if x is None:
            return np.zeros(self.fulldim)

        chars = len(x)
        output = []
        if len(x) > self.maxlen:
            x = x[:self.maxlen]

        imgs = []
        for i in x:
            imgs.append(self.get_embedding(i))

        return stitch_imgs(imgs, self.array_dims, self.dim, self.channels)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'embedding_filename': self.embedding_filename,
                'maxlen': self.maxlen,
                'zero_to_one': self.zero_to_one,
                'dim': self.dim
            }
        }


register_custom_object("VisualCharacters", VisualCharacters)
