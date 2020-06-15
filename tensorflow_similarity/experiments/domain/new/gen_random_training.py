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

""" Generates random training data for training a "domain" model.
We've since decided that this probably causes the model to learn the wrong thing.
"""

import tensorflow as tf
import multiprocessing
import strgen
from tqdm import tqdm
import numpy as np
import os

dir = os.path.dirname(__file__)
default_output_file = os.path.join(dir, "..", "data", "domains_notld.random")

flags.DEFINE_integer("N", 10000000, "")
flags.DEFINE_integer("length", 32, "")
flags.DEFINE_string("output_file", default_output_file, "")

FLAGS = flags.FLAGS


def gen_strings(x):
    np.random.seed(x)
    l = int(np.random.normal(16, 8))
    l = max(l, 8)
    l = min(l, 32)

    return strgen.StringGenerator("[0-9a-z-._]{%d}" % l).render_list(
        100, unique=True)


p = multiprocessing.Pool(64)

with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:

    # for strs in tqdm(p.imap_unordered(gen_strings, range(FLAGS.N / 100)),
    # total=(FLAGS.N/100)):
    pbar = tqdm(total=FLAGS.N)
    length_dist = [0] * 33
    for strs in p.imap_unordered(gen_strings, range(int(FLAGS.N / 100))):
        for s in strs:
            length_dist[len(s)] += 1
            f.write(s + "\n")
        pbar.update(100)

    print(length_dist)
