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
import collections
import glob
import io
import json
from tensorflow_similarity.experiments.domain.domain_augmentation import GetCharacterSpace
from tensorflow_similarity.preprocessing.visual_character_embedding import encode_img, decode_img
import multiprocessing
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import random
import six
import tensorflow as tf
from tqdm import tqdm
import zlib

flags.DEFINE_string("embeddings_directory", None,
                    "Directory containing ttf files.")
flags.DEFINE_string("output_file", None, "File.")
FLAGS = flags.FLAGS


def main(args):
    font_embeddings = glob.glob("%s/*.ttf" % FLAGS.embeddings_directory)

    last_shape = None
    combined_dict = collections.defaultdict(dict)
    for embedding in font_embeddings:
        font_key = os.path.basename(embedding)
        print("Reading %s" % embedding)
        with tf.io.gfile.GFile(embedding, "r") as f:
            d = json.load(f)
            for char, di in six.iteritems(d):
                for font, value in six.iteritems(di):
                    v = decode_img(value)
                    last_shape = np.shape(v)
                    combined_dict[char][font_key] = v

    shape = [len(combined_dict), len(font_embeddings)]
    shape.extend(list(last_shape))

    data = np.zeros(shape)

    char_space = sorted(list(combined_dict.keys()))
    for ki, key in enumerate(char_space):
        emb_dict = combined_dict[key]
        fonts = sorted(list(emb_dict.keys()))
        for ei, e_key in enumerate(fonts):
            emb = emb_dict[e_key]
            data[ki, ei] = emb

    encoded = encode_img(data)

    char_idx = {}
    for idx, c in enumerate(char_space):
        char_idx[c] = idx
    font_idx = {}
    for idx, f in enumerate(fonts):
        font_idx[f] = idx

    data = {
        'character_map': char_idx,
        'font_map': font_idx,
        'embeddings': encoded
    }

    with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    main(args)
