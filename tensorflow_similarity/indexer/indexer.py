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

class Indexer(object):
    """ Indexer class that indexes Embeddings. This allows for efficient
        searching of approximate nearest neighbors for a given embedding
        in metric space.
    """

    def __init__(self, dataset, model_path, index_dir):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
        self.dataset = dataset
        self.index_dir = index_dir
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.thresholds = dict()

    def build():
        """ build an index from a dataset 
        """
        # TODO
        pass

    def find(item, num_neighbors):
        """ find the closest data points and their associated data in the index

            Args:
                item (Item): The item for a which a query of the most similar items should be performed
                num_neighbors (int): The number of neighbors that should be returned

            Returns:
                neighbors (list(Item)): A list of the nearest neighbor items
        """
        # TODO
        pass

    def save(path):
        """ Store an indexer on the disk
        """
        # TODO
        pass

    def load(path):
        """ Load an indexer from the disk
        """
        # TODO
        pass
    
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

    def rebuild():
        """ Rebuild the index after updates were made
        """
        # TODO
        pass

    def compute_thresholds():
        """ Compute thresholds for similarity using V measure score
        """
        # TODO
        pass
