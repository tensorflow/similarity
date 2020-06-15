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
from tensorflow_similarity.utils.config_utils import register_custom_object
import tensorflow as tf
import numpy as np
import os
import random
from tensorflow_similarity.experiments.domain.new.homoglyphs import get_default_confusables
from tensorflow_similarity.experiments.domain.new.tlds import TLDS

ASCII_VOWELS = set(['a', 'e', 'i', 'o', 'u'])
ASCII_CONSONANTS = set(list("bcdfghjklmnpqrstvwxyz"))
ASCII_DIGITS = set(list("0123456789"))
ASCIIBET = ASCII_VOWELS.union(ASCII_CONSONANTS).union(ASCII_DIGITS)
CONFUSABLES = get_default_confusables()


def GetDomains(file):
    f = open(file)
    output = []
    for domain in f:
        domain = domain.strip()
        # The data has punycode strings. We need to get our string
        # into bytes format (which requires the ascii encoding) so we
        # can decode internationalized odmain names (Punycode / IDNA)
        output.append(domain.encode("ascii").decode("idna"))
    random.shuffle(output)
    return output


def random_transpose(text):
    l = len(text)
    if l == 2:
        return text[1] + text[0]

    position = np.random.choice(l - 1)

    l = text[position]
    r = text[position + 1]

    prefix = text[:position]
    suffix = text[position + 2:]
    return prefix + r + l + suffix


def random_roll(i):
    split_point = int(len(i) / 2)
    left = i[:split_point]
    right = i[split_point:]
    return right + left


def random_tld(i):
    last_dot_index = i.rfind(".")
    if last_dot_index >= 0:
        i = i[:last_dot_index]
    else:
        i = i + "."
    i += np.random.choice(TLDS)

    if len(i) > 32:
        i = i[:32]
    return i


def random_deletion(text):
    position = random.randint(0, len(text) - 1)
    prefix = text[:position]
    suffix = text[position + 1:]
    return prefix + suffix


def random_ascii_insertion(text):
    position = random.randint(0, len(text))
    new_char = random.choice(list(ASCIIBET))
    return __insert(text, position, new_char, replace=False)


def __insert(text, idx, val, replace=False):
    prefix = text[:idx]
    if replace:
        suffix = text[idx + 1:]
    else:
        suffix = text[idx:]
    out = prefix + val + suffix
    return out


def random_ascii_substitution(text):
    position = random.randint(0, len(text) - 1)
    c = text[position]
    if c in ASCII_VOWELS:
        options = ASCII_VOWELS.difference([c]).union(ASCII_DIGITS)
    elif c in ASCII_CONSONANTS:
        options = ASCII_CONSONANTS.difference([c]).union(ASCII_DIGITS)
    elif c in ASCII_DIGITS:
        options = ASCII_DIGITS.difference([c])
    else:
        options = ASCIIBET
    new_char = random.choice(list(options))
    return __insert(text, position, new_char, replace=True)


def random_confusable_substitution(text):
    position = random.randint(0, len(text) - 1)
    c = text[position]
    options = CONFUSABLES.get(c, None)
    if not options:
        options = ASCII_CONSONANTS.union(ASCII_VOWELS)
    new_char = random.choice(list(options))

    out = []
    chars = list(text)
    for cc in chars:
        if cc == c:
            out.append(new_char)
        else:
            out.append(cc)

    return ''.join(out)


def random_confusable_substitution_all(text):
    position = random.randint(0, len(text) - 1)
    c = text[position]

    out = []
    chars = list(text)
    for c in chars:
        options = CONFUSABLES.get(c, None)
        if not options:
            options = [c]
        out.append(random.choice(list(options)))
    return ''.join(out)


def random_repetition(text):
    position = random.randint(0, len(text) - 1)
    cnt = random.randint(1, max(int(len(text) / 4), 2))
    val = text[position] * cnt
    return __insert(text, position, val, replace=False)


def get_common_words():
    filename = os.path.join(os.path.dirname(__file__), "data", "common_words")
    with tf.io.gfile.GFile(filename) as f:
        return f.read().split("\n")


COMMON_ADDITIONS = get_common_words()

PREFIXES = ["www", "ssl"] + COMMON_ADDITIONS

SUFFIXES = ["s", "es", "enespanol"] + COMMON_ADDITIONS

JOINERS = ["", ".", "_", "-"]


def add_suffix(text):
    joiner = np.random.choice(JOINERS)
    s = np.random.choice(SUFFIXES)
    return text + joiner + s


def add_prefix(text):
    joiner = np.random.choice(JOINERS)
    s = np.random.choice(PREFIXES)
    return s + joiner + text


def insert_useless_punctuation(text):
    punc = ["-", "_", "."]
    position = random.randint(1, len(text) - 1)
    return __insert(text, position, np.random.choice(punc), replace=False)


SHORT_TEXT_AUGS = [random_ascii_insertion, random_ascii_substitution]

AUG_FREQUENCIES = [(insert_useless_punctuation, 1), (add_prefix, 1),
                   (add_suffix, 1), (random_tld, 1),
                   (random_confusable_substitution_all, 2),
                   (random_confusable_substitution, 3), (random_transpose, 3),
                   (random_ascii_insertion, 1), (random_ascii_substitution, 1),
                   (random_deletion, 1), (random_repetition, 1),
                   (random_roll, 1)]

ALL_AUGS = []
for aug, cnt in AUG_FREQUENCIES:
    ALL_AUGS.extend([aug] * cnt)


def augment(text, times=1):
    if not text:
        return ["", "N/A"]

    o = SHORT_TEXT_AUGS if len(text) <= 3 else ALL_AUGS
    tries = 3

    if times > 1:
        times = random.randint(1, times)

    output = text
    while tries and output == text:
        fs = []
        output = text
        for i in range(times):
            fn = np.random.choice(o)
            output = fn(output)
            fs.append(fn)
        tries = tries - 1

    return output, fs


class DomainAugment(object):
    def __init__(self, times=1, field="example"):
        self.times = times
        self.field = field

    def __call__(self, x):
        value = x[self.field]
        out = copy.copy(x)
        value = augment(value, times=self.times)[0]
        out[self.field] = value
        return out

    def get_config(self):
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "times": self.times
            }
        }


register_custom_object("DomainAugment", DomainAugment)

if __name__ == '__main__':

    #    aug = DomainAugment(1)
    for i in range(1000):
        print(random_confusable_substitution_all("google"))
