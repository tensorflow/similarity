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

import glob
import h5py
import numpy as np
import multiprocessing
import random
from tensorflow.keras.utils import HDF5Matrix
import tensorflow_similarity.utils.sample
from tqdm import tqdm


def read_file_pattern(file_pattern,
                      file_type='h5',
                      in_memory=True,
                      preprocess=None,
                      preprocessing_workers=None,
                      example_feature='x',
                      class_feature='y',
                      progress_bar=True):
    # This should be expanded in the future.
    if file_type == 'h5':
        return read_hdf5_file_pattern(
            file_pattern,
            in_memory=in_memory,
            preprocess=preprocess,
            example_group=example_feature,
            class_group=class_feature,
            progress_bar=progress_bar,
            preprocessing_workers=preprocessing_workers)
    else:
        raise NotImplementedError()


def write_hdf5_file(file, x, y, example_group='x', class_group='y'):
    with h5py.File(file, "w") as f:
        f.create_dataset(example_group, data=x)
        f.create_dataset(class_group, data=y)


def read_hdf5_file(file,
                   example_group='x',
                   class_group='y',
                   in_memory=True,
                   preprocess=None,
                   preprocessing_workers=None):
    if in_memory:
        with h5py.File(file, "r") as f:
            examples = f[example_group][:]
            classes = f[class_group][:]
            if preprocess:
                if preprocessing_workers:
                    with multiprocessing.Pool(preprocessing_workers) as pool:
                        examples = pool.map(preprocess, examples)
                else:
                    examples = [preprocess(x) for x in examples]
            return examples, classes
    else:
        return (HDF5Matrix(file, example_group, normalizer=preprocess),
                HDF5Matrix(file, class_group, normalizer=preprocess))


def open_hdf5_file_pattern(file_pattern, shuffle_order=True):
    files = glob.glob(file_pattern)
    if shuffle_order:
        random.shuffle(files)
    return [h5py.File(file, "r") for file in files]


def read_hdf5_file_pattern(file_pattern,
                           example_group='x',
                           class_group='y',
                           in_memory=True,
                           concat=True,
                           limit=None,
                           preprocess=None,
                           preprocessing_workers=None,
                           progress_bar=True):
    # If we're preprocessing, you have to accept that the preprocessed
    # values are being stored in memory.
    assert not (preprocess and not in_memory)

    files = glob.glob(file_pattern)
    random.shuffle(files)
    if limit:
        files = files[:min(limit, len(files))]

    xs = []
    ys = []

    if progress_bar:
        files = tqdm(files, unit='shard', desc='Loading %s' % file_pattern)

    for file in files:
        x, y = read_hdf5_file(
            file,
            in_memory=in_memory,
            preprocess=preprocess,
            preprocessing_workers=preprocessing_workers,
            example_group=example_group,
            class_group=class_group)
        xs.append(x)
        ys.append(y)

    if concat:
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    else:
        return xs, ys


def read_h5_file_pattern(file_pattern,
                         example_group='x',
                         class_group='y',
                         limit=None,
                         preprocess=None,
                         progress_bar=True):
    files = glob.glob(file_pattern)
    random.shuffle(files)
    if limit:
        files = files[:min(limit, len(files))]
    output = []
    if progress_bar:
        files = tqdm(files, desc="Files read")
    for file in files:
        x = HDF5Matrix(file, example_group, normalizer=preprocess)
        y = HDF5Matrix(file, class_group, normalizer=preprocess)
        output.append(x, y)
