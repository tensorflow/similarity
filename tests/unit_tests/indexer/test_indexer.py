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
import shutil
from scipy import spatial
import tensorflow as tf
import json
import math
import nmslib
import jsonlines
from tensorflow_similarity.indexer.indexer import Indexer
from tensorflow_similarity.indexer.utils import (load_packaged_dataset, read_json_lines, write_json_lines)


def set_up():
    """" Generate an indexer and a dataset
    """
    examples = np.random.randint(1000, size=(50, 400))
    labels = np.random.randint(2, size=50)
    _, tmp_file_examples = tempfile.mkstemp()
    with jsonlines.open(tmp_file_examples, mode='w') as writer:
        for data_point in examples:
            writer.write(data_point.tolist())
    _, tmp_file_labels = tempfile.mkstemp()
    with jsonlines.open(tmp_file_labels, mode='w') as writer:
        for data_point in labels:
            writer.write(data_point.tolist())
    temp_dir = tempfile.mkdtemp()

    dataset_examples_path = os.path.abspath(tmp_file_examples)
    dataset_labels_path = os.path.abspath(tmp_file_labels)
    model_path = os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/IMDB_model.h5")
    index_dir = temp_dir
    thresholds = {"likely":1}

    indexer = Indexer(dataset_examples_path, 
                      dataset_labels_path, 
                      model_path,
                      index_dir, 
                      thresholds=thresholds)

    return indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir

def delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir):
    """ Delete temporary files/directories that were generated as part of testing
    """
    shutil.rmtree(temp_dir)
    os.remove(tmp_file_examples)
    os.remove(tmp_file_labels)

def test_read_json_lines():
    """ Test case that asserts whether json lines reader util
        reads a json lines file correctly
    """
    arr = np.random.rand(400, 50).tolist()
    _, tmp_file = tempfile.mkstemp()
    with jsonlines.open(tmp_file, mode='w') as writer:
        for data_point in arr:
            writer.write(data_point)
    decoded_arr = read_json_lines(tmp_file)
    os.remove(tmp_file)

    assert(arr == decoded_arr)

def test_write_json_lines():
    """ Test case that asserts whether json lines writer util
        writes json lines files correctly
    """
    data = np.random.rand(400,)
    _, tmp_file = tempfile.mkstemp()
    write_json_lines(tmp_file, data)
    temp_data = []
    with open(tmp_file) as f:
        for line in f:
            temp_data.append(json.loads(line))
    os.remove(tmp_file)
    data = np.random.rand(400,50)
    _, tmp_file = tempfile.mkstemp()
    write_json_lines(tmp_file, data.tolist())
    temp_data = []
    with open(tmp_file) as f:
        for line in f:
            temp_data.append(json.loads(line))
    os.remove(tmp_file)

    assert((data == temp_data).all())
    assert((data == temp_data).all())

def test_load_packaged_dataset():
    """ Test case that asserts whether the data set loading util
        loads a saved dataset correctly
    """
    examples = np.random.rand(400, 50)
    labels = np.random.rand(400,)
    _, tmp_file_examples = tempfile.mkstemp()
    with jsonlines.open(tmp_file_examples, mode='w') as writer:
        for data_point in examples:
            writer.write(data_point.tolist())
    _, tmp_file_labels = tempfile.mkstemp()
    with jsonlines.open(tmp_file_labels, mode='w') as writer:
        for data_point in labels:
            writer.write(data_point.tolist())

    packaged_examples, packaged_labels = load_packaged_dataset(os.path.abspath(tmp_file_examples), 
                                                               os.path.abspath(tmp_file_labels), 
                                                               "test")
    os.remove(tmp_file_examples)
    os.remove(tmp_file_labels)

    assert((labels == packaged_labels).all())
    assert((examples == packaged_examples["test"]).all())

def test_build():
    """ Test case that asserts that the indexer correctly
        builds an index from a dataset
    """
    indexer, examples, labels, tmp_file_examples, tmp_file_labels, _ = set_up()
    indexer.build()
    ids, dists = indexer.index.knnQuery(examples[0], k=10)
    
    os.remove(tmp_file_examples)
    os.remove(tmp_file_labels)

    assert(isinstance(indexer.index, nmslib.dist.FloatIndex))
    assert(isinstance(ids, np.ndarray))
    assert(isinstance(dists, np.ndarray))

def test_find():
    """ Test case that asserts that the indexer correctly
        finds the most similar embeddings and their distances
    """
    data_set = np.asarray(read_json_lines(os.path.abspath("test_data_set/data.json")))

    dataset_examples_path = os.path.abspath("test_data_set/data.json")
    dataset_labels_path = os.path.abspath("test_data_set/labels.json")
    model_path = os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/IMDB_model.h5")
    index_dir = "./bundle"

    indexer = Indexer(dataset_examples_path, 
                      dataset_labels_path, 
                      model_path, 
                      index_dir)
    indexer.index.addDataPointBatch(data_set)
    indexer.index.createIndex()
    neighbors = indexer.find(data_set[0], 20, True)
    
    index_dists = np.asarray([neighbor.distance for neighbor in neighbors])
    index_ids = np.asarray([neighbor.id for neighbor in neighbors])
    dists = np.asarray([(spatial.distance.cosine(i, data_set[0])) for i in data_set[:20]])
    ids = np.arange(20)

    assert(np.isclose(index_dists, dists).all())
    assert((index_ids == ids).all())

def test_save():
    """ Test case that asserts that the indexer is correctly
        saved to the disk
    """
    indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = set_up()
    indexer.build()
    indexer.save()

    saved_examples= np.asarray(read_json_lines(os.path.abspath(os.path.join(temp_dir, "examples.jsonl"))))
    saved_labels = read_json_lines(os.path.abspath(os.path.join(temp_dir, "labels.jsonl")))

    saved_index = nmslib.init(method='hnsw', space="cosinesimil")
    saved_index.loadIndex(os.path.abspath(os.path.join(temp_dir, "index")), True)
    saved_index.createIndex()

    temp_model = tf.keras.models.load_model(os.path.join(os.path.abspath(temp_dir), "model.h5"))
    num = np.random.randint(1000, size=(1, 400))
    neighbors = indexer.find(num, 10)
    temp_ids, temp_dists = saved_index.knnQuery(temp_model.predict({'text': num}), 10)

    index_dists = np.asarray([neighbor.distance for neighbor in neighbors])
    index_ids = np.asarray([neighbor.id for neighbor in neighbors])
    
    delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

    assert((saved_examples== indexer.dataset_examples[indexer.model_dict_key]).all())
    assert((saved_labels == indexer.dataset_labels).all())
    assert((temp_ids == index_ids).all())
    assert((temp_dists == index_dists).all())

def test_load():
    """ Test case that asserts that a saved indexer correctly
        loads from the disk
    """
    indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = set_up()
    indexer.build()
    num = np.random.randint(1000, size=(1, 400))
    neighbors = indexer.find(num, 10)
    indexer.save()
    loaded_indexer = Indexer.load(os.path.abspath(temp_dir))
    
    delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

    assert((indexer.dataset_examples[indexer.model_dict_key] == loaded_indexer.dataset_examples[loaded_indexer.model_dict_key]).all())
    assert((indexer.dataset_labels == loaded_indexer.dataset_labels).all())
    assert(indexer.thresholds == loaded_indexer.thresholds)

def test_add():
    """ Test case that asserts that the indexer correctly
        adds new items to the indexer
    """
    indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = set_up()
    indexer.build()
    num = np.random.randint(1000, size=(1, 400))
    examples = np.concatenate((examples, num))
    labels = np.append(labels, 0)
    indexer.add(num, 0)
    
    delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)
    
    assert((examples== indexer.dataset_examples[indexer.model_dict_key]).all())
    assert((labels == indexer.dataset_labels).all())

def test_remove():
    """ Test case that asserts that the indexer correctly
        removes items from the indexer
    """
    indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = set_up()
    indexer.build()
    indexer.remove(0)
    
    delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

    assert((indexer.dataset_examples[indexer.model_dict_key] == examples[1:]).all())
    assert((indexer.dataset_labels == labels[1:]).all())
    indexer.remove(len(indexer.dataset_labels) - 1)
    assert((indexer.dataset_examples[indexer.model_dict_key] == examples[1:-1]).all())

def test_compute_threhsolds():
    """ Test case that asserts that the indexer correctly
        computes the thresholds for similarity
    """
    indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = set_up()
    indexer.build()
    indexer.compute_thresholds()
    
    delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

    assert(indexer.thresholds[.1] == "possible")
