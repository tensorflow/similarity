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
import collections
import unittest
from scipy import spatial
import tensorflow as tf
import json
from mock import patch
import math
import nmslib
import jsonlines
from tensorflow_addons.losses import TripletHardLoss
from tensorflow_similarity.indexer.indexer import Indexer
from tensorflow_similarity.indexer.utils import (load_packaged_dataset,
                                                 read_json_lines,
                                                 write_json_lines)

Neighbor = collections.namedtuple("Neighbor",
                                  ["id",
                                   "data",
                                   "distance",
                                   "label"])
NUM_CALIBRATION_NEIGHBORS = 5
NUM_CALIBRATION_EXAMPLES = 4


class IndexerTestCase(unittest.TestCase):

    def generate_mock(self):
        """ Generate a list of neighbors for mocking
        """
        neighbors = []
        for i in range(NUM_CALIBRATION_EXAMPLES):
            neighbor_list = []
            for j in range(NUM_CALIBRATION_NEIGHBORS):
                # id and data do not matter for calibration
                neighbor_id = np.int32(j)
                neighbor_data = np.asarray([i, j])
                # Generate increasing distances for each nearest
                # neighbor list
                neighbor_distance = np.float32(i / 16 + j / 25)

                # Generate NUM_CALIBRATION_EXAMPLES - 1 correct labels
                # and 1 incorrect label
                if j < NUM_CALIBRATION_EXAMPLES:
                    neighbor_label = np.int64(i + 1)
                else:
                    neighbor_label = np.int64(i)

                neighbor_mock = Neighbor(id=neighbor_id,
                                        data=neighbor_data,
                                        distance=neighbor_distance,
                                        label=neighbor_label)

                neighbor_list.append(neighbor_mock)

            neighbors.append(neighbor_list)

        return neighbors

    def set_up(self):
        """" Generate an indexer and a dataset
        """
        # Generate dataset
        examples = np.random.rand(50, 28, 28)
        labels = np.asarray([0,1] * 25)

        # Write examples to temp file
        _, tmp_file_examples = tempfile.mkstemp()
        with jsonlines.open(tmp_file_examples, mode='w') as writer:
            for data_point in examples:
                writer.write(data_point.tolist())

        # Write labels to temp file
        _, tmp_file_labels = tempfile.mkstemp()
        with jsonlines.open(tmp_file_labels, mode='w') as writer:
            for data_point in labels:
                writer.write(data_point.tolist())
        temp_dir = tempfile.mkdtemp()

        dataset_examples_path = os.path.abspath(tmp_file_examples)
        dataset_labels_path = os.path.abspath(tmp_file_labels)
        model_path = os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/MNIST_model")
        index_dir = temp_dir

        indexer = Indexer(dataset_examples_path,
                        dataset_labels_path,
                        model_path)

        return indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir


    def delete_temp_files(self, tmp_file_examples, tmp_file_labels, temp_dir):
        """ Delete temporary files/directories that were generated as part of testing
        """
        shutil.rmtree(temp_dir)
        os.remove(tmp_file_examples)
        os.remove(tmp_file_labels)


    def test_read_json_lines(self):
        """ Test case that asserts whether json lines reader util
            reads a json lines file correctly
        """
        # Generate dataset
        arr = np.random.rand(400, 50).tolist()

        # Write data to temp file
        _, tmp_file = tempfile.mkstemp()
        with jsonlines.open(tmp_file, mode='w') as writer:
            for data_point in arr:
                writer.write(data_point)

        decoded_arr = read_json_lines(tmp_file)
        os.remove(tmp_file)

        assert(arr == decoded_arr)


    def test_write_json_lines(self):
        """ Test case that asserts whether json lines writer util
            writes json lines files correctly
        """
        # Generate dataset labels
        data = np.random.rand(400,)
        _, tmp_file = tempfile.mkstemp()
        write_json_lines(tmp_file, data)

        # read dataset labels
        temp_data = []
        with open(tmp_file) as f:
            for line in f:
                temp_data.append(json.loads(line))
        os.remove(tmp_file)

        # Generate dataset examples
        data = np.random.rand(400, 50)
        _, tmp_file = tempfile.mkstemp()
        write_json_lines(tmp_file, data.tolist())

        # read dataset labels
        temp_data = []
        with open(tmp_file) as f:
            for line in f:
                temp_data.append(json.loads(line))
        os.remove(tmp_file)

        assert((data == temp_data).all())
        assert((data == temp_data).all())


    def test_load_packaged_dataset(self):
        """ Test case that asserts whether the data set loading util
            loads a saved dataset correctly
        """
        # Generate dataset
        examples = np.random.rand(400, 50)
        labels = np.random.rand(400,)

        # Write dataset examples to temp file
        _, tmp_file_examples = tempfile.mkstemp()
        with jsonlines.open(tmp_file_examples, mode='w') as writer:
            for data_point in examples:
                writer.write(data_point.tolist())

        # Write dataset labels to temp file
        _, tmp_file_labels = tempfile.mkstemp()
        with jsonlines.open(tmp_file_labels, mode='w') as writer:
            for data_point in labels:
                writer.write(data_point.tolist())

        # Load dataset as packaged dataset from examples temp file and labels temp file
        packaged_examples, packaged_labels = load_packaged_dataset(os.path.abspath(tmp_file_examples),
                                                                os.path.abspath(tmp_file_labels))
        os.remove(tmp_file_examples)
        os.remove(tmp_file_labels)

        assert((labels == packaged_labels).all())
        assert((examples == packaged_examples).all())


    def test_build(self):
        """ Test case that asserts that the indexer correctly
            builds an index from a dataset
        """
        # Build an indexer and query it for 10 nearest neighbors
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, _ = self.set_up()
        indexer.build()
        ids, dists = indexer.index.knnQuery(examples[0], k=10)

        os.remove(tmp_file_examples)
        os.remove(tmp_file_labels)

        assert(isinstance(indexer.index, nmslib.dist.FloatIndex))
        assert(isinstance(ids, np.ndarray))
        assert(isinstance(dists, np.ndarray))


    def test_single_embedding_find(self):
        """ Test case that asserts that the indexer correctly
            finds the most similar embeddings and their distances
            for an item that is in embedding form
        """
        dataset_examples_path = os.path.abspath("test_data_set/data.json")
        dataset_labels_path = os.path.abspath("test_data_set/labels.json")
        model_path = os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/MNIST_model")

        # Load the dataset from the test_data_set directory
        data_set = np.asarray(read_json_lines(dataset_examples_path))

        # Build an indexer and find the 10 nearest neighbors of the first
        # embedding in the dataset
        indexer = Indexer(dataset_examples_path,
                        dataset_labels_path,
                        model_path)
        indexer.index.addDataPointBatch(data_set)
        indexer.index.createIndex()
        indexer.lookup_time = 0
        indexer.num_lookups = 0
        neighbors = indexer.find(np.asarray([data_set[0]]), 10, True)[0]

        # Get the ids and distances for the queried nearest neighbors
        index_dists = np.asarray([neighbor.distance for neighbor in neighbors])
        index_ids = np.asarray([neighbor.id for neighbor in neighbors])

        # Get the ids and distances for the 10 closest embeddings in the dataset
        dists = np.asarray([(spatial.distance.cosine(i, data_set[0])) for i in data_set[:10]])
        ids = np.arange(10)

        assert(np.isclose(index_dists, dists).all())
        assert((index_ids == ids).all())


    def test_multiple_examples_find(self):
        """ Test case that asserts that the indexer correctly
            finds the most similar embeddings and their distances
            for multiple items that are not in embedding form
        """
        # Build an in indexer
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, tmp_dir = self.set_up()
        indexer.build()

        # Generate multiple examples and query the indexer for the nearest neighbors
        examples = np.random.rand(1000, 28, 28)
        neighbors = indexer.find(items=examples,
                                num_neighbors=20,
                                is_embedding=False)

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, tmp_dir)

        neighbors_sorted = True
        for neighbor_list in neighbors:
            for i in range(len(neighbor_list) - 1):
                if neighbor_list[i].distance > neighbor_list[i + 1].distance:
                    neighbors_sorted = False

        assert(neighbors_sorted)


    def test_save(self):
        """ Test case that asserts that the indexer is correctly
            saved to the disk
        """
        # Build an indexer and save it to disk
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = self.set_up()
        indexer.build()
        indexer.save(index_dir=temp_dir)

        # Load the saved dataset examples
        saved_examples_path = os.path.abspath(os.path.join(temp_dir, "examples.jsonl"))
        saved_examples = np.asarray(read_json_lines(saved_examples_path))

        # Load the saved dataset labels
        saved_labels_path = os.path.abspath(os.path.join(temp_dir, "labels.jsonl"))
        saved_labels = read_json_lines(saved_labels_path)

        # Load the saved index
        saved_index = nmslib.init(method='hnsw', space="cosinesimil")
        saved_index.loadIndex(os.path.abspath(os.path.join(temp_dir, "index")), True)
        saved_index.createIndex()

        # Load the saved model
        saved_model_path = os.path.join(os.path.abspath(temp_dir), "model")
        saved_model = tf.keras.models.load_model(saved_model_path, 
                                                custom_objects={
                                                    'tf': tf, 
                                                    'TripletHardLoss': TripletHardLoss
                                                })

        # Generate a datapoint and use the loaded model to produce an embedding for it
        num = np.random.randint(1000, size=(1, 28, 28))
        neighbors = indexer.find(num, 10)[0]
        embedding = saved_model.predict(num)

        # Query the loaded index for the 10 nearest neighbors of the embedding
        temp_ids, temp_dists = saved_index.knnQuery(embedding, 10)

        # Query the index for the 10 nearest neighbors of embedding
        index_dists = np.asarray([neighbor.distance for neighbor in neighbors])
        index_ids = np.asarray([neighbor.id for neighbor in neighbors])

        indexer_dataset_examples = indexer.dataset_examples
        indexer_dataset_labels = indexer.dataset_labels

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

        assert((saved_examples == indexer_dataset_examples).all())
        assert((saved_labels == indexer_dataset_labels).all())
        assert((temp_ids == index_ids).all())
        assert((temp_dists == index_dists).all())


    def test_load(self):
        """ Test case that asserts that a saved indexer correctly
            loads from the disk
        """
        # Build an indexer
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = self.set_up()
        indexer.build()

        # Save the indexer to disk and load it
        indexer.save(index_dir=temp_dir)
        loaded_indexer = Indexer.load(os.path.abspath(temp_dir))

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

        indexer_dataset_examples = indexer.dataset_examples
        loaded_indexer_dataset_examples = loaded_indexer.dataset_examples

        indexer_dataset_labels = indexer.dataset_labels
        loaded_indexer_dataset_labels = loaded_indexer.dataset_labels

        assert((indexer_dataset_examples == loaded_indexer_dataset_examples).all())
        assert((indexer_dataset_labels == loaded_indexer_dataset_labels).all())


    def test_add(self):
        """ Test case that asserts that the indexer correctly
            adds new items to the index
        """
        # Build an indexer
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = self.set_up()
        indexer.build()

        # Get the class distribution and the number of embeddings in the indexer
        class_distribution = indexer.compute_class_distribution()
        num_embeddings = indexer.compute_num_embeddings()

        # Generate a datapoint and add it to the dataset examples and dataset labels
        num = np.random.rand(28, 28)
        examples = np.concatenate((examples, np.asarray([num])))
        labels = np.append(labels, 0)
        
        class_distribution[0] = class_distribution[0] + 1
        num_embeddings = num_embeddings + 1

        # Add the datapoint to the indexer
        indexer.add([num], [0])

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

        indexer_dataset_examples = indexer.dataset_examples
        indexer_dataset_labels = indexer.dataset_labels
        indexer_num_embeddings = indexer.compute_num_embeddings()

        assert((examples == indexer_dataset_examples).all())
        assert((labels == indexer_dataset_labels).all())
        assert(class_distribution == indexer.compute_class_distribution())
        assert(num_embeddings == indexer_num_embeddings)


    def test_remove(self):
        """ Test case that asserts that the indexer correctly
            removes items from the indexer
        """
        # Build an indexer
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = self.set_up()
        indexer.build()

        # Get the class distribution and the number of embeddings in the indexer
        class_distribution = indexer.compute_class_distribution()
        num_embeddings = indexer.compute_num_embeddings()

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

        # Remove the first datapoint in the dataset
        indexer.remove([0])
        indexer_dataset_examples = indexer.dataset_examples
        indexer_dataset_labels = indexer.dataset_labels
        label = labels[0]
        class_distribution[label] = class_distribution[label] - 1
        num_embeddings = num_embeddings - 1

        assert((indexer_dataset_examples == examples[1:]).all())
        assert((indexer_dataset_labels == labels[1:]).all())
        assert(indexer.compute_class_distribution() == class_distribution)
        assert(num_embeddings == indexer.compute_num_embeddings())

        # Remove the last datapoint in the dataset
        indexer.remove([len(indexer.dataset_labels) - 1])
        indexer_dataset_examples = indexer.dataset_examples
        indexer_dataset_labels = indexer.dataset_labels
        label = labels[len(labels) - 1]
        class_distribution[label] = class_distribution[label] - 1
        num_embeddings = num_embeddings - 1

        assert((indexer_dataset_examples == examples[1:-1]).all())
        assert((indexer_dataset_labels == labels[1:-1]).all())
        assert(indexer.compute_class_distribution() == class_distribution)
        assert(num_embeddings == indexer.compute_num_embeddings())


    def test_get_info(self):
        """ Test case that asserts that the indexer correctly
            returns information about the data it stores.
        """
        # Build an indexer and get information about embedding size
        # and number of embeddigns
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = self.set_up()
        indexer.build()
        info = indexer.get_info()
        num_embeddings = info["num_embeddings"] 
        embedding_size = info["embedding_size"]
        class_distribution = info["class_distribution"]

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

        assert(num_embeddings == len(examples))
        assert(embedding_size == 16)
        assert(class_distribution[0] == 25)
        assert(class_distribution[1] == 25)


    def test_get_metrics(self):
        """ Test case that asserts that the indexer correctly
            returns performance metrics.
        """
        # Build an indexer and get performance metrics
        indexer, examples, labels, tmp_file_examples, tmp_file_labels, temp_dir = self.set_up()
        indexer.build()
        metrics = indexer.get_metrics()
        avg_query_time = metrics["avg_query_time"]
        num_lookups = metrics["num_lookups"]

        self.delete_temp_files(tmp_file_examples, tmp_file_labels, temp_dir)

        assert(num_lookups == 0)
        assert(avg_query_time == 0)

        # Generate multiple examples and query the indexer for the nearest neighbors
        examples = np.random.rand(25, 28, 28)
        _ = indexer.find(items=examples,
                        num_neighbors=20,
                        is_embedding=False)

        # Get performance metrics
        metrics = indexer.get_metrics()
        avg_query_time = metrics["avg_query_time"]
        num_lookups = metrics["num_lookups"]

        assert(num_lookups == 25)
        assert(avg_query_time >= 0)


    def test_calibration(self):
        """ Test case that asserts that the indexer correctly
            performs calibration.
        """
        dataset_examples_path = os.path.abspath("test_data_set/data.json")
        dataset_labels_path = os.path.abspath("test_data_set/labels.json")
        model_path = os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/MNIST_model")
        indexer = Indexer(dataset_examples_path,
                        dataset_labels_path,
                        model_path)

        # Load the dataset from the test_data_set directory
        examples, labels = load_packaged_dataset(dataset_examples_path, dataset_labels_path)

        # Build an indexer and perform calibration
        with patch.object(Indexer, "find", return_value=self.generate_mock()) as indexer_find_mock:
            calibration = indexer.calibrate(examples[:NUM_CALIBRATION_EXAMPLES],
                                            labels[:NUM_CALIBRATION_EXAMPLES],
                                            num_neighbors=NUM_CALIBRATION_NEIGHBORS)

        thresholds = calibration['thresholds']
        threshold_distances = np.asarray(thresholds['distance'])
        thresholds_precision = thresholds['precision']
        thresholds_recall = thresholds['recall']
        thresholds_f1 = thresholds['f1']
        binary_threshold = calibration['binary_threshold']

        true_precision = np.asarray([
            1.0, 0.8, 0.83, 0.86, 0.88, 0.89, 0.8, 0.82,
            0.83, 0.85, 0.86, 0.8, 0.81, 0.82, 0.83, 0.84, 0.8])

        true_recall = np.asarray([0.25, 0.25, 0.31, 0.38, 0.44,
            0.5, 0.5, 0.56, 0.62, 0.69, 0.75, 0.75, 0.81, 0.88, 
            0.94, 1.0, 1.0])

        true_f1 = np.asarray([0.4, 0.38095238095238093,
            0.45454545454545453, 0.5217391304347825,
            0.5833333333333334, 0.64, 0.6153846153846154,
            0.6666666666666666, 0.7142857142857143,
            0.7586206896551724, 0.7999999999999999,
            0.7741935483870969, 0.8125, 0.8484848484848485,
            0.8823529411764706, 0.9142857142857143,
            0.888888888888889])

        true_distances = np.asarray([0.08, 0.1025, 0.12,
            0.125, 0.1425, 0.16, 0.165, 0.1825, 0.1875,
            0.205, 0.2225, 0.2275, 0.245, 0.2675, 0.285,
            0.3075, 0.3475]).astype('float32')

        assert(binary_threshold == np.float32(0.3075))
        assert((thresholds_precision == true_precision).all())
        assert((thresholds_recall == true_recall).all())
        assert((thresholds_f1 == true_f1).all())
        assert((threshold_distances == true_distances).all())


    def test_compute_labels(self):
        """ Test case that asserts that the indexer correctly
            computes similarity labels.
        """
        dataset_examples_path = os.path.abspath("test_data_set/data.json")
        dataset_labels_path = os.path.abspath("test_data_set/labels.json")
        model_path = os.path.abspath("../../../tensorflow_similarity/serving/www/saved_models/MNIST_model")
        indexer = Indexer(dataset_examples_path,
                        dataset_labels_path,
                        model_path)

        # Load the dataset from the test_data_set directory
        examples, labels = load_packaged_dataset(dataset_examples_path, dataset_labels_path)

        # Build an indexer and perform calibration
        with patch.object(Indexer, "find", return_value=self.generate_mock()) as indexer_find_mock:
            calibration = indexer.calibrate(examples[:NUM_CALIBRATION_EXAMPLES],
                                            labels[:NUM_CALIBRATION_EXAMPLES],
                                            num_neighbors=NUM_CALIBRATION_NEIGHBORS)
        
        distances = np.asarray(calibration['thresholds']['distance'])
        precisions = calibration['thresholds']['precision']

        # Compute the similarity labels
        labels = indexer.compute_labels(precisions, distances)

        true_labels = {
            'very_likely': np.float32(0.08),
            'likely': np.float32(0.1025),
            'possible': -1
        }

        assert(labels == true_labels)
