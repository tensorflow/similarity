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

import collections
from tensorflow.keras.callbacks import Callback
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm


def run(pool, fn, value):
    if pool is not None:
        return pool.map(fn, value)
    else:
        out = []
        for x in value:
            out.append(fn(x))
        return out


def preprocess(reader, preprocessing, writer, num_workers=4, verbose=True):
    input_dict = reader.read()

    if num_workers is not None:
        pool = multiprocessing.Pool(num_workers)
    else:
        pool = None

    data_dict = {}
    str_data_dict = {}

    items = input_dict.items()
    if verbose:
        items = tqdm(items, desc="Preprocessing")

    for entry in items:
        name, (x, y) = entry
        tqdm.write("Prepocessing %s" % name)
        logging.getLogger("debug").debug("Preprocessing '%s'" % name)

        x_pp = np.array(run(pool, preprocessing, x))

        # Special handling for string/unicode outputs, since varlen items
        # need to be handled specially.
        if x_pp.dtype.char == 'S' or x_pp.dtype.char == 'U':
            str_data_dict['x_%s' % name] = x_pp
        else:
            data_dict['x_%s' % name] = x_pp

        if y.dtype.char == 'S' or y.dtype.char == 'U':
            str_data_dict['y_%s' % name] = y
        else:
            data_dict['y_%s' % name] = y

    if pool is not None:
        pool.close()

    writer.write(data_dict, str_data_dict)
