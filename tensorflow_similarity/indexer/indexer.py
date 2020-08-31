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
import itertools
import os
import time
import collections
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

        indexer = cls(dataset_examples_path=dataset_examples_path,
                      dataset_original_path=dataset_original_path,
                      dataset_labels_path=dataset_labels_path,
                      model_path=model_path)
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


    def __compute_calibration_metrics(self, class_matches):
        """ Calculate calibration Precision at R, Recall and F1

            Args:
                class_matches (np.ndarray): A list of bool indicating whether the nearest
                                            neighbors for each calibration example is the
                                            same class.

            Returns:
                precision_scores (list(float)): A list containing the precisions at each 
                                                point in sorted_distances.
                recall_scores (list(float)): A list containing the recall at each point 
                                             in sorted_distances.
                f1_scores (list(float)): A list containing the f1 score at each point in 
                                         sorted_distances.
        """
        correct_matches = 0
        precision_scores = []
        recall_scores = []
        f1_scores = []
        count = 0

        # Find the total number of matches
        num_total_match = sum(class_matches)

        # Compute r precision, recall and f1
        for class_match in class_matches:
            count += 1

            if class_match:
                correct_matches += 1
        
            # Calculate precision at pos, recall at pos,
            # and f1 at pos
            precision = correct_matches / count
            recall = correct_matches / num_total_match
            f1 = (precision * recall  / (precision + recall)) * 2

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        return precision_scores, recall_scores, f1_scores


    def __compute_calibration_thresholds(
        self,
        sorted_distances,
        precision_scores,
        recall_scores,
        f1_scores,
        decimals
    ):
        """ Calculate thresholds and normalized similarity labels

            Args:
                sorted_distances (np.ndarray): A list of distances between nearest neighbors 
                                               and calibration examples sorted in ascending order.
                precision_scores (list(float)): A list containing the precisions at each 
                                                point in sorted_distances.
                recall_scores (list(float)): A list containing the recall at each point 
                                             in sorted_distances.
                f1_scores (list(float)): A list containing the f1 score at each point in 
                                         sorted_distances.
                decimals (int): The number of decimals to use when rounding computed metrics.

            Returns:
                thresholds (dict): A dict containing a list of thresholds, as well as 
                                   precision, recall and f1 scores at each threshold.
                binary_threshold (float): The saddle point of the F1 curve. This 
                                          thresholds splits the calibration results
                                          into the same class and not the same class.

        """
        thresholds = defaultdict(list)
        rows = []
        curr_precision = 1.1
        num_distances = len(sorted_distances)

        # Normalize the labels and compute thresholds
        for idx in range(num_distances - 1, -1, -1):
            # Don't round f1 or distance because we need the numerical
            # precision for precise boundary computations
            f1 = f1_scores[idx]
            distance = sorted_distances[idx]
            
            # Round precision and recall
            precision = round(precision_scores[idx], decimals)
            recall = round(recall_scores[idx], decimals)

            # Perform threshold binning
            if precision != curr_precision:
                thresholds['precision'].append(precision)
                thresholds['recall'].append(recall)
                thresholds['f1'].append(f1)
                thresholds['distance'].append(distance)
                curr_precision = precision

        # Compute the optimal thresholding distance which is the 
        # saddle point on the f1 curve
        binary_threshold = thresholds['distance'][np.argmax(thresholds['f1'])]

        # Reverse the metrics and distances in thresholds since they are currently in 
        # ascending order 0.01, 0.02,... -> 0.99, 0.98,... so the best thresholds
        # appear first
        for v in thresholds.values():
            v.reverse()

        return thresholds, binary_threshold


    def calibrate(
        self, 
        examples, 
        labels, 
        decimals=2,
        num_neighbors=10,
    ):
        """ Calibrate indexer and compute threshold distances for similarity.
            Calibration reduces multi class classification to binary classification
            and requires a reasonably good model to work properly.

            Args:
                examples (np.ndarray): The examples that calibration should be
                                       performed on.
                labels (np.ndarray): The labels corresponding to the examples.
                num_neighbors (int): The number of neighbors that should be used for calibration.
                                     Defaults to 10.

            Returns:
                calibration (dict): A dictionary containing a dictionary of distance, f1 scores,
                                    precision scores and recall scores at each threshold, and the 
                                    binary threshold.
        """
        # Query the index for the nearest neighbors
        neighbors = self.find(examples, num_neighbors=num_neighbors)
        flatten = itertools.chain.from_iterable
        flattened_neighbors = list(flatten(neighbors))

        # Get distances for all nearest neighbors
        distances = list(map(lambda neighbor: neighbor.distance, flattened_neighbors))
        sorted_distances = sorted(distances)

        # Get labels for all nearest neighbors
        neighbor_labels = list(map(lambda neighbor: neighbor.label, flattened_neighbors))

        # Get the nearest neighbors where the label is correct
        true_labels = np.repeat(labels, num_neighbors)
        class_matches = true_labels == neighbor_labels

        #  Compute r precision, f1 and recall for all neighbors
        precision_scores, recall_scores, f1_scores = self.__compute_calibration_metrics(class_matches)
    
        # Compute similarity thresholds and normalize labels
        thresholds, binary_threshold = self.__compute_calibration_thresholds(sorted_distances,
                                                                                     precision_scores,
                                                                                     recall_scores,
                                                                                     f1_scores,
                                                                                     decimals)

        calibration = {
            "binary_threshold": binary_threshold,
            "thresholds": thresholds,
        }

        return calibration


    def compute_labels(
        self,
        precisions,
        distances,
        label_thresholds,
        decimals=2
    ):
        """ Compute similarity labels and their thresholdss

            Args:
                precisions (list(float)): A list containing the precisions at each 
                                          point in distances.
                distances (np.ndarray): A list of distances between nearest neighbors 
                                        and calibration examples sorted in ascending order.
                label_thresholds (dict): A dictionary of precisions mapping to their respective
                                         thresholds. In the example {0.9: 'very_likely',0.8: 'likely'}
                                         items would be considered very_likely to be similar
                                         at precision 0.9.
                decimals (int): The number of decimals to use when rounding computed metrics.
                                Defaults to 2.

            Returns:
                labels (dict): A dict containing similarity labels mapping to their 
                               corresponding distance threshold. If the a label maps to -1
                               there exists no distance that lies in the distance threshold
                               range for that label.
        """
        # Initialize thresholds for all labels as -1
        labels = {v: -1 for _, v in label_thresholds.items()}
        curr_precision = 1.1

        # Find the smallest distance in each threshold range
        for idx in range(len(distances) - 1, -1, -1):
            distance = distances[idx]
            
            # Round precision
            precision = round(precisions[idx], decimals)

            # Update label threshold
            if precision != curr_precision:
                # Get the highest threshold value in label_thresholds less than precision
                max_threshold = max(k for k in label_thresholds if k <= precision)

                # Get the label associated with the threshold value
                max_threshold_label = label_thresholds[max_threshold]

                # Update labels
                labels[max_threshold_label] = distance
                curr_precision = precision

        return labels
