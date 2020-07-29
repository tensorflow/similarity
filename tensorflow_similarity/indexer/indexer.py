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
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow_similarity.indexer.utils import (load_packaged_dataset, read_json_lines, write_json_lines, write_json_lines_dict)

class Indexer(object):
    """ Indexer class that indexes Embeddings. This allows for efficient
        searching of approximate nearest neighbors for a given embedding
        in metric space.

        Args:
            model_path (string): The path to the model that should be used to calculate embeddings
            dataset_examples_path (string): The path to the json lines file containing the dataset
            dataset_original_path (string): The path to the json lines file containing the original dataset 
            dataset_labels_path (string): The path to the json lines file containing the labels for the dataset
            index_dir (string): The path to the directory where the indexer should be saved,
            space (string): The space (a space is a combination of data and the distance) to use in the indexer
                            for a list of available spaces see: https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
    """

    def __init__(self, dataset_examples_path, dataset_original_path, dataset_labels_path, model_path, index_dir, space="cosinesimil", thresholds=dict()):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
        self.dataset_examples, self.dataset_labels = load_packaged_dataset(dataset_examples_path, dataset_labels_path, self.model.layers[0].name)
        if dataset_original_path is not None:
            self.dataset_original = np.asarray(read_json_lines(dataset_original_path))
        else:
            self.dataset_original = self.dataset_examples[self.model.layers[0].name]
        self.index_dir = index_dir
        self.index = nmslib.init(method='hnsw', space=space)
        self.thresholds = thresholds

    def build(self, verbose=0):
        """ build an index from a dataset 

            Args:
                verbose (int): Verbosity mode (0 = silent, 1 = progress bar)
        """
        embeddings = self.model.predict(self.dataset_examples)
        _ = self.index.addDataPointBatch(embeddings)
        print_progess = verbose > 0
        self.index.createIndex(print_progress=print_progess)

    def find(self, item, num_neighbors, embedding=False):
        """ find the closest data points and their associated data in the index

            Args:
                item (np.array): The item for a which a query of the most similar items should be performed
                num_neighbors (int): The number of neighbors that should be returned
                embedding (bool): Whether or not the item is already in embedding form

            Returns:
                neighbors (np.array(dict)): A list of the nearest neighbor items
        """
        if not embedding:
            item = self.model.predict({self.model.layers[0].name: item})
        ids, dists = self.index.knnQuery(item, num_neighbors)
        neighbors = []
        for id, dist in zip(ids, dists):
            neighbors.append({"id": id, "data": self.dataset_original[id], "distance": dist, "label": self.dataset_labels[id]})
        return np.asarray(neighbors)

    def save(self):
        """ Store an indexer on the disk
        """
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        self.index.saveIndex(os.path.join(self.index_dir, "index"), True)
        write_json_lines(os.path.join(self.index_dir, "examples.json"), self.dataset_examples[self.model.layers[0].name].tolist())
        write_json_lines_dict(os.path.join(self.index_dir, "thresholds.json"), self.thresholds)
        write_json_lines(os.path.join(self.index_dir, "labels.json"), self.dataset_labels.tolist())
        write_json_lines(os.path.join(self.index_dir, "original_examples.json"), self.dataset_original.tolist())
        self.model.save(os.path.join(self.index_dir, "model.h5"))

    @classmethod
    def load(cls, path):
        """ Load an indexer from the disk

            Args:
                The path that the indexer should be loaded from
        """
        indexer = cls(dataset_examples_path=os.path.join(path, "examples.json"), dataset_original_path=os.path.join(path, "original_examples.json"), dataset_labels_path=os.path.join(path, "labels.json"), model_path=os.path.join(path, "model.h5"), index_dir="./bundle", thresholds=read_json_lines(os.path.join(path, "thresholds.json"))[0])
        indexer.index.loadIndex(os.path.join(path, "index"), True)
        indexer.rebuild()
        return indexer

    
    def add(item):
        """ Add an item to the index
        
            Args:
                item (Item): The item to be added to the index
        """
        # TODO
        pass

    def remove(item):
        """ Remove an item from the index
            Args:
                item (Item): The item to removed added to the index
        """
        # TODO
        pass

    def rebuild(self):
        """ Rebuild the index after updates were made
        """
        self.index.createIndex()

    def compute_thresholds(self):
        """ Compute thresholds for similarity using R Precision
        """
        data = []
        for embedding, label in zip(self.model.predict(self.dataset_examples), self.dataset_labels):
            ids, distances = self.index.knnQuery(embedding, len(self.dataset_labels))
            relevant = 0
            retrieved = 0
            data_point_thresholds = dict()
            for id, distance in zip(ids, distances):
                if self.dataset_labels[id] == label:
                    relevant = relevant + 1
                retrieved = retrieved + 1
                r_precision = round(relevant / retrieved, 2)
                data_point_thresholds[r_precision] = distance
            for thresh, dist in data_point_thresholds.items():
                l = self.thresholds.get(thresh, list())
                l.append(dist)
                self.thresholds[thresh] = l
        for threshold, threshold_list in self.thresholds.items():
            self.thresholds[threshold] = np.mean(threshold_list)

        sort_orders = sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True)

        for i in sort_orders:
            print(i[0], i[1])
