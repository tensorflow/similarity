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
import collections
from termcolor import cprint
import numpy as np
from collections import Counter
import tensorflow as tf
from nltk import ngrams
import multiprocessing
from tqdm import tqdm

flags.DEFINE_string("input", "/mnt/ramdisk/domains.critical", "")
flags.DEFINE_integer("num_words", 10000,
                     "Number of most frequently seen words to keep.")
FLAGS = flags.FLAGS

MAX_CHAR = 10
MIN_CHAR = 3
RANGE = MAX_CHAR - MIN_CHAR


def extract_ngrams(chunk):
    output = [[]] * RANGE

    for idx, size in enumerate(range(MIN_CHAR, MAX_CHAR)):
        counter = Counter()
        for line in chunk:
            line = line.strip()
            l = len(line)
            if size >= l:
                break

            for ngram in ngrams(line, size):
                counter[''.join(ngram)] += 1
        output[idx] = counter.most_common(128)
    return output


def words(chunk):
    counter = Counter()
    for line in chunk:
        line = line.replace(".", " ").replace("-", " ").replace("_", " ")
        line = line.strip()
        tokens = line.split(" ")
        counter.update(tokens)
    return counter.most_common(FLAGS.num_words)


with tf.io.gfile.GFile(FLAGS.input) as f:
    lines = f.readlines()

WORKERS = 71
SHARDS = 1024

pool = multiprocessing.Pool(WORKERS)
ngram_array = []

pbar = tqdm(total=SHARDS, desc="Extract NGrams")

counter = Counter()

results = collections.defaultdict(list)
# for shard in pool.imap_unordered(extract_ngrams, np.array_split(lines,
#                                                                 SHARDS)):
#     for i in range(RANGE):
#         results[i].extend(shard[i])
#     pbar.update(1)
# pbar.close()

results = []
for shard in pool.imap_unordered(words, np.array_split(lines, SHARDS)):
    results.extend(shard)
    pbar.update(1)
pbar.close()

counter = Counter(results)
for res in results:
    counter[res[0]] += res[1]

with tf.io.gfile.GFile("common_words", "w") as f:
    for item in counter.most_common(FLAGS.num_words):
        if item[1] > 1 and len(item[0]) >= MIN_CHAR:
            f.write("%s\n" % item[0])
    for item, cnt in counter.most_common(100):
        print(item, ":", cnt)

# with tf.io.gfile.GFile("common_grams", "w") as f:
#     for i in range(RANGE):
#         c = Counter()

#         for res in results[i]:
#             c[res[0]] += res[1]
#         for item in c.most_common(1000):
#             if item[1] > 1 and len(item[0]) >= MIN_CHAR:
#                 f.write("%s\n" % item[0])
#         print(c.most_common(20))
