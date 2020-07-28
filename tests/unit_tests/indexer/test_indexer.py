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

import pytest
import numpy as np
import os
import tempfile
import nmslib
from scipy import spatial
import json
import math
import nmslib
import jsonlines
from tensorflow_similarity.indexer.indexer import Indexer
from tensorflow_similarity.indexer.utils import (load_packaged_dataset, read_json_lines, write_json_lines)

def test_read_json_lines():
    arr = np.random.rand(400, 50).tolist()
    _, tmp_file = tempfile.mkstemp()
    with jsonlines.open(tmp_file, mode='w') as writer:
        for data_point in arr:
            writer.write(data_point)
    decoded_arr = read_json_lines(tmp_file)
    os.remove(tmp_file)
    assert(arr == decoded_arr)

def test_load_packaged_dataset():
    x = np.random.rand(400, 50)
    y = np.random.rand(400,)
    _, tmp_file_examples = tempfile.mkstemp()
    with jsonlines.open(tmp_file_examples, mode='w') as writer:
        for data_point in x:
            writer.write(data_point.tolist())
    _, tmp_file_labels = tempfile.mkstemp()
    with jsonlines.open(tmp_file_labels, mode='w') as writer:
        for data_point in y:
            writer.write(data_point.tolist())

    packaged_x, packaged_y = load_packaged_dataset(os.path.abspath(tmp_file_examples), os.path.abspath(tmp_file_labels), "test")
    os.remove(tmp_file_examples)
    os.remove(tmp_file_labels)
    assert((y == packaged_y).all())
    assert((x == packaged_x["test"]).all())

def test_write_json_lines():
    data = np.random.rand(400,)
    _, tmp_file = tempfile.mkstemp()
    write_json_lines(tmp_file, data)
    temp_data = []
    with open(tmp_file) as f:
        for line in f:
            temp_data.append(json.loads(line))
    os.remove(tmp_file)
    assert((data == temp_data).all())
    data = np.random.rand(400,50)
    _, tmp_file = tempfile.mkstemp()
    write_json_lines(tmp_file, data.tolist())
    temp_data = []
    with open(tmp_file) as f:
        for line in f:
            temp_data.append(json.loads(line))
    os.remove(tmp_file)
    assert((data == temp_data).all())

def test_build():
    x = np.random.randint(1000, size=(50, 400))
    y = np.random.randint(2, size=50)
    _, tmp_file_examples = tempfile.mkstemp()
    with jsonlines.open(tmp_file_examples, mode='w') as writer:
        for data_point in x:
            writer.write(data_point.tolist())
    _, tmp_file_labels = tempfile.mkstemp()
    with jsonlines.open(tmp_file_labels, mode='w') as writer:
        for data_point in y:
            writer.write(data_point.tolist())
    indexer = Indexer(os.path.abspath(tmp_file_examples), None, os.path.abspath(tmp_file_labels), os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/IMDB_model.h5"), "./")
    indexer.build()
    ids, dists = indexer.index.knnQuery(x[0], k=10)
    os.remove(tmp_file_examples)
    os.remove(tmp_file_labels)
    assert(isinstance(indexer.index, nmslib.dist.FloatIndex))
    assert(isinstance(ids, np.ndarray))
    assert(isinstance(dists, np.ndarray))

def test_find():
    data_set = np.asarray(read_json_lines(os.path.abspath("test_data_set/data.json")))
    indexer = Indexer(os.path.abspath("test_data_set/data.json"), None, os.path.abspath("test_data_set/labels.json"), os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/IMDB_model.h5"), "./")
    indexer.index.addDataPointBatch(data_set)
    indexer.index.createIndex()
    neighbors = indexer.find(data_set[0], 20, True)
    index_dists = np.asarray([neighbor["distance"] for neighbor in neighbors])
    index_ids = np.asarray([neighbor["id"] for neighbor in neighbors])
    dists = np.asarray([(spatial.distance.cosine(i, data_set[0])) for i in data_set[:20]])
    ids = np.arange(20)
    assert(np.isclose(index_dists, dists).all())
    assert((index_ids == ids).all())


def test_save():
    assert(True)

def test_load():
    assert(True)

def test_compute_threhsolds():
    assert(True)
