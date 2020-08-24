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
                               i.e. {.001: "very likely", .01: "likely", .1: "possible", .2: "unlikely"}

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
        self.model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
        self.model_dict_key = self.model.layers[0].name
        self.dataset_examples, self.dataset_labels = load_packaged_dataset(dataset_examples_path,
                                                                           dataset_labels_path,
                                                                           self.model_dict_key)
        self.space = space
        self.index = nmslib.init(method='hnsw', space=self.space)

        if dataset_original_path is not None:
            self.dataset_original = np.asarray(read_json_lines(dataset_original_path))
        else:
            self.dataset_original = self.dataset_examples[self.model_dict_key]

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
            items = self.model.predict({self.model_dict_key: items})

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
        dataset_examples  = self.dataset_examples[self.model_dict_key]
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
        self.model.save(os.path.join(index_dir, "model.h5"))


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
        model_path = os.path.join(path, "model.h5")
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
            if example.shape == self.dataset_examples[self.model_dict_key][0].shape:
                example = np.asarray([example])
            dataset_examples = np.concatenate((self.dataset_examples[self.model_dict_key], example))
            self.dataset_examples = {self.model_dict_key: dataset_examples}
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
        dataset_examples = np.delete(self.dataset_examples[self.model_dict_key], tuple(ids), 0)
        self.dataset_examples = {self.model_dict_key: dataset_examples}
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


    def compute_thresholds(self):
        """ Compute thresholds for similarity using R Precision.
        """
        # Currently the thresholds are placeholder values, in the future the indexer
        # will use R precision to calculate thresholds
        self.thresholds = {.001: "very likely", .01: "likely", .1: "possible", .2: "unlikely"}
        num_samples = 100
        rng = default_rng()
        sample = rng.choice(len(self.dataset_labels), size=num_samples, replace=False)
        sample.sort()

        pb = tqdm(total=num_samples ** 2, desc='Computing pairwise distances', unit='pairwise distances')
        
        distances = []
        # compute pairwise distances
        for id in sample:
            pairwise_distances = []
            for id_2 in sample:
                pairwise_distances.append(self.index.getDistance(id, id_2))
                pb.update()
            distances.append(pairwise_distances)
        pb.close()

        distances = np.asarray(distances)

        print(distances.shape)

        pb = tqdm(total=num_samples ** 2, desc='Computing class matches', unit='class matches')

        class_matches = []
        
        for x in sample:
            for y in sample:
                if self.dataset_labels[x] == self.dataset_labels[y]:
                    same_class = True
                else:
                    same_class = False
                class_matches.append(same_class)
                pb.update()
        pb.close()
        print("Class matches:", np.sum(class_matches), 'Total:', len(class_matches))

        #get all the class matching distance and sort them low to high
        flat_distances = np.reshape(distances, (num_samples * num_samples))
        matching_idxes = np.argwhere(class_matches)

        pb = tqdm(total=len(flat_distances), desc='Computing match ratios', unit='match ratios')

        match_rate = 0
        match_rate_ratios = []
        sorted_distance_values = []
        for pos, idx in enumerate(np.argsort(flat_distances)):
            distance_value = flat_distances[idx]

            # remove distance with self
            if not round(distance_value, 4):
                continue

            if idx in matching_idxes:
                match_rate += 1
            match_rate_ratios.append(match_rate/ (pos + 1))
            sorted_distance_values.append(distance_value)
            pb.update()
        pb.close()
        
        print(sorted_distance_values[:10])

        # plot to make sure everyting is fine
        # ratio should be about ~10% cause 10 dims
        plt.plot(match_rate_ratios)
        plt.title("match ratios")
        plt.savefig('mr.png')
        # distance should be between >0 and 1
        plt.title("distance values")
        plt.plot(sorted_distance_values)
        plt.savefig('dv.png')

        current_threshold = 1
        precision = 2  # 2 -> 0.01
        num_ratios = len(match_rate_ratios)
        threshold_ratios = []
        threshold_values = []
        for idx in range(num_ratios):
            neg_idx = num_ratios - idx - 1
            ratio = round(match_rate_ratios[neg_idx], precision)
            #print(neg_idx, ratio, current_threshold)
            # passing  the bar
            if ratio < current_threshold:
                current_threshold = ratio
                threshold_ratios.append(ratio)
                threshold_values.append(sorted_distance_values[idx])

        for idx, val in enumerate(threshold_values):
            print(idx, threshold_ratios[idx], val)

        print("Num ratios", num_ratios)
        print("match rate", match_rate)
        print("match rate ratios", match_rate_ratios[:10])