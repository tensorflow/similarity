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
import multiprocessing
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageDraw, ImageFont
import six
from tqdm import tqdm
import zlib

flags.DEFINE_string("validation_domains_file", None,
                    "Domain validation to be read.")
flags.DEFINE_integer("font_img_size", 8,
                     "Size of the rendered images to use.")
flags.DEFINE_integer("font_size", 8, "Size of the rendered images to use.")
flags.DEFINE_string("font_directory", None,
                    "Directory containing ttf files.")
flags.DEFINE_string("embedding_directory", "/tmp/",
                    "Directory to store embedding data.")

FLAGS = flags.FLAGS


def process_chunk(args):
    font_key, font, chunk = args
    output = collections.defaultdict(dict)

    for chr in chunk:
        img = Image.new("L", (FLAGS.font_img_size, FLAGS.font_img_size))
        d = ImageDraw.Draw(img)

        fnt = ImageFont.truetype(font, FLAGS.font_size)
        d.text((0, 0), chr, font=fnt, fill=(255, ))
        arr = np.array(img, dtype=np.uint8)

        f = io.BytesIO()
        np.save(f, arr)
        compressed = base64.b64encode(zlib.compress(
            f.getvalue())).decode("ascii")

        output[chr][font_key] = compressed
    return output


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(args):
    space = set(GetCharacterSpace())
    with tf.io.gfile.GFile(FLAGS.validation_domains_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            for c in line:
                space.add(c)
    space = list(space)

    chunks = np.array_split(space, 1000)

    print("Done.")

    p = multiprocessing.Pool(60)

    fonts = glob.glob("%s/*.ttf" % FLAGS.font_directory)

    with tqdm(total=(len(chunks) * len(fonts))) as pbar:
        for idx, font in enumerate(fonts):
            combined_dict = collections.defaultdict(dict)
            tqdm.write("Font: %s (%d/ %d)" % (font, idx, len(fonts)))
            font_key = os.path.basename(font)

            f_chunks = [[font_key, font, chunk] for chunk in chunks]
            for d in p.imap_unordered(process_chunk, f_chunks):
                for k, v in six.iteritems(d):
                    combined_dict[k] = v
                pbar.update(1)
            output_file = "%s/vis_embed_%s" % (FLAGS.embedding_directory,
                                               font_key)
            with tf.io.gfile.GFile(output_file, "w") as f:
                json.dump(combined_dict, f, cls=NumpyEncoder)


if __name__ == '__main__':
    main(args)
