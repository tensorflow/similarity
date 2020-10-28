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
import h5py
from tensorflow_similarity.experiments.domain.new.augment import *
import multiprocessing
import numpy as np
import os
import random
import sys
import tensorflow as tf
from tqdm import tqdm

DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIR, "data")

flags.DEFINE_string("popular_domains", os.path.join(DATA_DIR,
                                                    "domains.popular"), "")
flags.DEFINE_string("domain_pair_file",
                    os.path.join(DATA_DIR, "domains.ads-validation"), "")
flags.DEFINE_string("validation_domains",
                    os.path.join(DATA_DIR, "domains.validation"), "")
flags.DEFINE_string("critical_domains",
                    os.path.join(DATA_DIR, "domains.critical"), "")

flags.DEFINE_string("output_file", os.path.join(DATA_DIR, "golden_set_5"), "")
flags.DEFINE_integer("workers", 1, "")

FLAGS = flags.FLAGS


def get_validation_domains(f):
    aug = []
    real = []
    o = tf.io.gfile.GFile(f, "r")
    l = 0
    for line in o.readlines():
        l += 1
        line = line.strip()
        tokens = line.split("\t")
        if len(tokens) < 2:
            print("Bad line %s" % line)
            continue

        augmented_domain = tokens[0]
        real_domain = tokens[-1]

        augmented_domain = augmented_domain.split('.')[0]
        real_domain = real_domain.split('.')[0]

        aug.append(augmented_domain)
        real.append(real_domain)

    return real, aug


def add_augments(data_dict, name, fn, number, x, y):
    xout = []
    yout = []
    zout = []

    for i in range(number):
        xout.append(fn(x))
        yout.append(y)
        zout.append(name)
    return xout, yout, zout


def n_augmentations(n):
    def fn(x):
        # Drop the function calls.
        return augment(x, times=n)[0]

    return fn


def filter_domains(arr):
    o = []
    for i in arr:
        if len(i) > 5:
            o.append(i)
    return o


def main(args):
    critical = GetDomains(FLAGS.critical_domains)
    popular = filter_domains(GetDomains(FLAGS.popular_domains))
    random_validation = filter_domains(GetDomains(FLAGS.popular_domains))

    validation_targets, augmentations = get_validation_domains(
        FLAGS.domain_pair_file)

    aug_domains = list(set(popular[:250] + critical))
    validation_domains = list(set(aug_domains + validation_targets))

    domain_class_map = {}
    for i, d in enumerate(validation_domains):
        domain_class_map[d] = i

    validation_sets = [[], [], []]

    for domain in validation_domains:
        validation_sets[0].append(domain)
        validation_sets[1].append(domain_class_map[domain])
        validation_sets[2].append("targets")

    for target, augment in zip(validation_targets, augmentations):
        validation_sets[0].append(augment)
        validation_sets[1].append(domain_class_map[target])
        validation_sets[2].append("ads_validation")

    single_outputs = {
        "insert_useless_punctuation": insert_useless_punctuation,
        "add_prefix": add_prefix,
        "add_suffix": add_suffix,
        "random_confusable_substitution_all":
        random_confusable_substitution_all,
        "random_confusable_substitution": random_confusable_substitution,
        "random_transpose": random_transpose,
        "random_ascii_insertion": random_ascii_insertion,
        "random_ascii_substitution": random_ascii_substitution,
        "random_deletion": random_deletion,
        "random_repetition": random_repetition,
        "single_manipulations": n_augmentations(1),
        "dual_manipulations": n_augmentations(2),
        "triple_manipulations": n_augmentations(3),
        "quad_manipulations": n_augmentations(4)
    }

    sets = list(single_outputs.keys())

    for raw_domain in aug_domains:
        cls = domain_class_map[raw_domain]

        for name, fn in single_outputs.items():
            x_, y_, z_ = add_augments(validation_sets, name, fn, 5, raw_domain,
                                      cls)
            validation_sets[0].extend(x_)
            validation_sets[1].extend(y_)
            validation_sets[2].extend(z_)

    with h5py.File(FLAGS.output_file, "w") as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("subsets", (len(sets), ), dtype=dt)
        f['subsets'][:] = sets

        examples = validation_sets[0]
        labels = validation_sets[1]
        groups = validation_sets[2]

        f.create_dataset("examples", (len(examples), ), dtype=dt)
        f["examples"][:] = examples

        f.create_dataset("labels", data=labels)

        f.create_dataset("groups", (len(groups), ), dtype=dt)
        f["groups"][:] = groups


if __name__ == '__main__':
    multiprocessing.freeze_support()
    app.run(main)
