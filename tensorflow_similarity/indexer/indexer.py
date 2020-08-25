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

import nmslib
from collections import defaultdict
import tensorflow as tf
from tensorflow_addons.losses import TripletHardLoss
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import os
import time
import collections
from tqdm.auto import tqdm
from tensorflow_similarity.indexer.utils import (load_packaged_dataset,
                                                 read_json_lines,
                                                 write_json_lines,
                                                 write_json_lines_dict)

# Neighbor has the distance from the queried point to the item, the index
# of that item in the dataset, the label of the item and the data in the
# dataset associated with the item.
Neighbor = collections.namedtuple("Neighbor", ["id", "data", "distance", "label"])


class Indexer(object):
    """ Indexer class that indexes Embeddings. This allows for efficient
        searching of approximate nearest neighbors for a given embedding
        in metric space.

        Args:
            dataset_examples_path (string): The path to the json lines file containing the dataset that
                                            should be indexed and is ingestible by the model.
            dataset_labels_path (string): The path to the json lines file containing the labels for the dataset
            model_path (string): The path to the model that should be used to calculate embeddings
            dataset_original_path (string): The path to the json lines file containing the original dataset.
                                            The original dataset should be used for datasets where the original
                                            data is not ingestible by the model or the raw data differs from the
                                            dataset examples. The original dataset is used to visualize datapoints
                                            for datasets where the original data cannot be reconstructed from the
                                            data ingested by the model such as text datasets. Defaults to None.
            space (string): The space (a space is a combination of data and the distance) to use in the indexer
                            for a list of available spaces see: https://github.com/nmslib/nmslib/blob/master/manual/spaces.md.
                            Defaults to "cosinesimil".
            thresholds (dict): A dictionary mapping likeliness labels to thresholds. Defaults to None.
                               i.e. {.001: "very likely", .01: "likely", .1: "possible"}

        Attributes:
            embedding_size (int): The size of the embeddings stored by the indexer.
            num_lookups (int): The number of lookups performed by the indexer.
            lookup_time (float): The cumulative amount of time it took to perform lookups.
    """

    def __init__(
        self,
        dataset_examples_path,
        dataset_labels_path,
        model_path,
        dataset_original_path=None,
        space="cosinesimil",
        thresholds=None
    ):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf, 'TripletHardLoss': TripletHardLoss})
        self.dataset_examples, self.dataset_labels = load_packaged_dataset(dataset_examples_path,
                                                                           dataset_labels_path)
        self.space = space
        self.index = nmslib.init(method='hnsw', space=self.space)

        if dataset_original_path is not None:
            self.dataset_original = np.asarray(read_json_lines(dataset_original_path))
        else:
            self.dataset_original = self.dataset_examples

        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = dict()


    def build(self, verbose=0, rebuild_index=False, loaded_index=False):
        """ build an index from a dataset

            Args:
                verbose (int): Verbosity mode (0 = silent, 1 = progress bar).
                               Defaults to 0.
                rebuild_index (bool): Whether to rebuild the index. Defaults to False.
                loaded_index (bool): Whether the index was loaded from disk.
                                     Defaults to False.
        """
        # Compute the embeddings for the dataset examples and add them to the index
        if rebuild_index:
            self.index = nmslib.init(method='hnsw', space=self.space)

        embeddings = self.model.predict(self.dataset_examples)
        self.embedding_size = len(embeddings[0])

        if not loaded_index:
            self.index.addDataPointBatch(embeddings)

        print_progess = verbose > 0
        self.index.createIndex(print_progress=print_progess)
        self.num_lookups = 0
        self.lookup_time = 0


    def find(self, items, num_neighbors, is_embedding=False):
        """ find the closest data points and their associated data in the index

            Args:
                items (list): The items for which a query of the most similar items should
                              be performed
                num_neighbors (int): The number of neighbors that should be returned
                is_embedding (bool): Whether or not the item is already in embedding form.
                                     Defaults to False.

            Returns:
                output (list(list(Neighbor))): A list of nearest neighbor items lists
                                               sorted by distance for each item that the
                                               query was performed on.
        """
        items = np.asarray(items)

        if not is_embedding:
            items = self.model.predict(items)

        # Query the index
        start_time = time.time()
        neighbors = self.index.knnQueryBatch(items, num_neighbors)
        query_time = time.time() - start_time

        self.lookup_time = self.lookup_time + query_time

        output = []
        for (ids, distances) in neighbors:
            query_neighbors = []
            for id, distance in zip(ids, distances):
                neighbor = Neighbor(id=id,
                                    data=self.dataset_original[id],
                                    distance=distance,
                                    label=self.dataset_labels[id])
                query_neighbors.append(neighbor)
            output.append(query_neighbors)
            self.num_lookups = self.num_lookups + 1

        return output


    def save(self, index_dir):
        """ Store an indexer on the disk
            index_dir (string): The path to the directory where the indexer
                                should be saved.
        """
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        # Save the index
        self.index.saveIndex(os.path.join(index_dir, "index"), True)

        # Save the examples
        examples_path = os.path.join(index_dir, "examples.jsonl")
        dataset_examples  = self.dataset_examples
        write_json_lines(examples_path, dataset_examples)

        # Save the labels
        labels_path = os.path.join(index_dir, "labels.jsonl")
        dataset_labels = self.dataset_labels
        write_json_lines(labels_path, dataset_labels)

        # Save the original dataset
        original_examples_path = os.path.join(index_dir, "original_examples.jsonl")
        dataset_original_examples = self.dataset_original
        write_json_lines(original_examples_path, dataset_original_examples)

        # Save the thresholds
        thresholds_path = os.path.join(index_dir, "thresholds.jsonl")
        write_json_lines_dict(thresholds_path, self.thresholds)

        # Save the model
        self.model.save(os.path.join(index_dir, "model"), save_format="tf2")


    @classmethod
    def load(cls, path):
        """ Load an indexer from the disk

            Args:
                path (string): The path that the indexer should be loaded from disk.
                               This directory should contain a tf.similarity model, dataset and
                               label files in jsonl format, a NMSLib index, and a
                               thresholds dictionary.
        """
        dataset_examples_path = os.path.join(path, "examples.jsonl")
        dataset_original_path = os.path.join(path, "original_examples.jsonl")
        dataset_labels_path = os.path.join(path, "labels.jsonl")
        model_path = os.path.join(path, "model")
        thresholds = read_json_lines(os.path.join(path, "thresholds.jsonl"))[0]

        indexer = cls(dataset_examples_path=dataset_examples_path,
                      dataset_original_path=dataset_original_path,
                      dataset_labels_path=dataset_labels_path,
                      model_path=model_path,
                      thresholds=thresholds)
        indexer.index.loadIndex(os.path.join(path, "index"), True)
        indexer.index.createIndex()
        indexer.build(loaded_index=True)

        return indexer


    def add(self, examples, labels, original_examples=None):
        """ Add item(s) to the index

            Args:
                example (list(np.array)): A list of the items to be added to the index.
                label (integer): A list of the labels corresponding to the items.
                original_example (object): A list of the original data points if different from examples.
                                           Defaults to None.
            Returns:
                ids (list(int)): A list of the ids of the items added to the index
        """
        if not original_examples:
            original_examples = [None] * len(labels)

        ids = []

        for example, label, original_example in zip(examples, labels, original_examples):
            # Add the example to the dataset examples, dataset labels, the original dataset,
            # and the class distribution, and rebuild the index
            if example.shape == self.dataset_examples[0].shape:
                example = np.asarray([example])
            dataset_examples = np.concatenate((self.dataset_examples, example))
            self.dataset_examples =  dataset_examples
            self.dataset_labels = np.append(self.dataset_labels, label)
            if original_example:
                self.dataset_original = np.concatenate((self.dataset_original, original_example))
            else:
                self.dataset_original = np.concatenate((self.dataset_original, example))
            ids.append(len(self.dataset_labels) - 1)

        self.build(rebuild_index=True)

        return ids


    def remove(self, ids):
        """ Remove item(s) from the index

            Args:
                ids (list(int)): A list of indeces of the items in the dataset to be 
                                 removed from the index.
        """
        # Delete the item from the dataset examples, original dataset, the dataset labels
        # and the class distribution, and rebuild the index
        dataset_examples = np.delete(self.dataset_examples, tuple(ids), 0)
        self.dataset_examples = dataset_examples
        self.dataset_original = np.delete(self.dataset_original, tuple(ids), 0)
        self.dataset_labels = np.delete(self.dataset_labels, tuple(ids))
        self.build(rebuild_index=True)


    def compute_class_distribution(self):
        """ Compute how many instances of each class are stored in the indexer

            Rerturns:
                class_distribution (dict): A dictionary mapping labels to the number of 
                                           examples with that label in the indexer.
        """
        # Get the counts for each class
        classes, counts = np.unique(self.dataset_labels, return_counts=True)
        class_distribution = defaultdict(int)

        # Convert to a JSON serializable dictionary
        for label, count in zip(classes.tolist(), counts.tolist()):
            class_distribution[label] = count

        return class_distribution


    def compute_num_embeddings(self):
        """ Compute the number of embeddings stored by the indexer

            Returns:
                (int): The number of embeddings sotred by the indexer.
        """
        return len(self.dataset_labels)


    def get_info(self):
        """ Get information about the data stored by the indexer

            Returns:
                dict: A dict containing the number of embeddings stored by the indexer and
                      the size of the embeddings stored by the indexer.
        """
        class_distribution = self.compute_class_distribution()
        num_embeddings = self.compute_num_embeddings()

        return {
            "num_embeddings": num_embeddings,
            "embedding_size": self.embedding_size,
            "class_distribution": class_distribution
        }


    def get_metrics(self):
        """ Get performance metrics from the indexer

            Returns:
                dict: A dict containing the number of lookups performed by the indexer and the
                      average time taken per query
        """
        if self.num_lookups > 0:
            avg_query_time = self.lookup_time / self.num_lookups
        else:
            avg_query_time = self.lookup_time

        return {
            "num_lookups": self.num_lookups,
            "avg_query_time": avg_query_time
        }
    

    def calibrate(
        self, 
        examples, 
        labels, 
        very_likely_threshold=0.9, 
        likely_threshold=0.8,
        possible_threshold=0.7, 
        metric_rounding=2
    ):
        """ Calibrate indexer and compute threshold distances for similarity

            Args:
                examples (np.ndarray): The examples that calibration should be
                                       performed on.
                labels (np.ndarray): The labels corresponding to the examples.
                very_likely_threshold (float): The threshold for which items should be
                                     considered very likely to be similar.
                                     Should be between 0 and 1. Defaults to 0.9.
                likely_threshold (float): The threshold for which items should be considered 
                                likely to be similar. Should be between 0 and 1. 
                                Defaults to 0.9.
                possible_threshold (float): The threshold for which items should be considered
                                  to possibly be similar. Should be between 0 and 1.
                                  Defaults to 0.1.
                metric_rounding (int): The number of decimals to use when rounding 
                                       computed metrics. Defaults to 2.
        """
        distances = []
        same_class = []
        label_idx = 0
        progress_bar = tqdm(total=len(examples) * 10,
                            desc='Computing class matches',
                            unit='class matches')

        # Query the index for the nearest neighbors
        neighbors = self.find(examples, num_neighbors=10, is_embedding=False)

        # Compute class matches and distances
        for neighbor_list in neighbors:
            for neighbor in neighbor_list:
                if neighbor.label == labels[label_idx]:
                    same_class.append(True)
                else:
                    same_class.append(False)
                distances.append(neighbor.distance)
                progress_bar.update()
            label_idx += 1
        progress_bar.close()
        
        # Find the indeces of all matches and the total number of matches
        matching_idxes = np.argwhere(same_class)
        num_total_match = sum(same_class)

        match_rate = 0
        precision_scores = []
        recall_scores = []
        f1_scores = []
        sorted_distance_values = []
        count = 0

        # Compute r precision, recall and f1
        for pos, idx in enumerate(np.argsort(distances)):
            distance_value = distances[idx]

            # Remove distance with self
            if not round(distance_value, 4):
                continue
                
            count += 1
            if idx in matching_idxes:
                match_rate += 1
            precision = match_rate / (count)
            recall = match_rate / num_total_match
            f1 = (precision * recall  / (precision + recall)) * 2

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            sorted_distance_values.append(distance_value)

        thresholds = defaultdict(list)
        rows = []
        curr_precision = 100000

        labels = {}
        num_distances = len(sorted_distance_values)

        # Normalize the labels and compute thresholds
        for ridx in range(num_distances):
            idx = num_distances - ridx - 1
            f1 = f1_scores[idx] 
            distance = sorted_distance_values[idx]  
            
            precision = round(precision_scores[idx], metric_rounding)
            recall = round(recall_scores[idx], metric_rounding)

            if precision != curr_precision:
                thresholds['precision'].append(precision)
                thresholds['recall'].append(recall)
                thresholds['f1'].append(f1)
                thresholds['distance'].append(distance)
                curr_precision = precision

                if precision >= very_likely_threshold:
                    labels['very_likely'] = distance
                elif precision >= likely_threshold:
                    labels['likely'] = distance
                elif precision >= possible_threshold:
                    labels['possible'] = distance

        # Compute the optimal thresholding distance
        binary_threshold = thresholds['distance'][np.argmax(thresholds['f1'])]

        for v in thresholds.values():
            v.reverse()

        calibration = {
            "binary_threshold": binary_threshold,
            "thresholds": thresholds,
            "labels": labels
        }

        return calibration