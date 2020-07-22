import nmslib
import tensorflow as tf
from tensorflow_similarity.indexer.utils import (load_packaged_dataset)

class Indexer(object):
    """ Indexer class that indexes Embeddings. This allows for efficient
        searching of approximate nearest neighbors for a given embedding
        in metric space.

        Args:
            model_path (string): The path to the model that should be used to calculate embeddings
            dataset (string): The path to the json lines file containing the dataset
            dataset_labels (string): The path to the json lines file containing the labels for the dataset
            index_dir (string): The path to the directory where the indexer should be saved,
            space (string): The space (a space is a combination of data and the distance) to use in the indexer
                            for a list of available spaces see: https://github.com/nmslib/nmslib/blob/master/manual/spaces.md
    """

    def __init__(self, dataset, dataset_labels, model_path, index_dir, space="cosinesimil"):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
        self.dataset, self.dataset_labels = load_packaged_dataset(dataset, dataset_labels, self.model.layers[0].name)
        self.index_dir = index_dir
        self.index = nmslib.init(method='hnsw', space=space)
        self.thresholds = dict()

    def build(self):
        """ build an index from a dataset 
        """
        embeddings = self.model.predict(self.dataset)
        self.index.addDataPointBatch(embeddings)
        self.index.createIndex(print_progress=True)

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

    def save():
        """ Store an indexer on the disk
        """
        # TODO
        pass

    def load(path):
        """ Load an indexer from the disk

            Args:
                The path to the file that the indexer is be loaded from
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
