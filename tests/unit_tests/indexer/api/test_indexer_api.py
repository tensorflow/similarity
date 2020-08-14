from fastapi.testclient import TestClient
from tensorflow_similarity.indexer.api.main import app
import numpy as np
from tensorflow_similarity.indexer.indexer import Indexer
import unittest
from mock import patch
import collections
import json


client = TestClient(app)

Neighbor = collections.namedtuple("Neighbor",
                                  ["id",
                                   "data",
                                   "distance",
                                   "label"])


class IndexerTestCase(unittest.TestCase):

    def generate_mock():
        neighbor_id = np.int32(1)
        neighbor_data = np.asarray([0])
        neighbor_distance = np.float32(.234)
        neighbor_label = np.int64(0)
        neighbor_mock = [Neighbor(id=neighbor_id,
                                  data=neighbor_data,
                                  distance=neighbor_distance,
                                  label=neighbor_label)]

        return neighbor_mock

    @patch.object(Indexer, 'find', return_value=[generate_mock()])
    def test_lookup_embeddings(self, Indexer):
        """ Test case that asserts that the API correctly performs a lookup
            of the nearest neighbors for a list of embeddings
        """
        # Generate embeddings
        embeddings = np.random.uniform(low=-1.0, high=0.99, size=(2, 4))
        embedding_list = embeddings.tolist()
        num_neighbors = 1

        # Query the API for the nearest neighbors of the embeddings
        response = client.post(
            "/lookupEmbeddings",
            json={"num_neighbors": num_neighbors, "embeddings": embedding_list}
        )

        # Convert the response to a namedtuple
        response_item = response.json()[0]
        id = response_item['id']
        label = response_item['label']
        data = response_item['data']
        distance = response_item['distance']

        response_neighbor = Neighbor(id=id,
                                     data=data,
                                     label=label,
                                     distance=distance)

        neighbor = Neighbor(id=np.int32(1),
                            data=np.asarray([0]),
                            distance=np.float32(0.234),
                            label=np.int64(0))

        assert(neighbor == response_neighbor)

    @patch.object(Indexer, 'find', return_value=[generate_mock()])
    def test_lookup(self, Indexer):
        """ Test case that asserts that the API correctly performs a lookup
            of the nearest neighbors for a list of examples
        """
        # Generate embeddings
        embeddings = np.random.randint(1000, size=(2, 28))
        embedding_list = embeddings.tolist()
        num_neighbors = 1

        # Query the API for the nearest neighbors of the embeddings
        response = client.post(
            "/lookup",
            json={"num_neighbors": num_neighbors, "embeddings": embedding_list}
        )

        # Convert the response to a namedtuple
        response_item = response.json()[0]
        id = response_item['id']
        label = response_item['label']
        data = response_item['data']
        distance = response_item['distance']

        response_neighbor = Neighbor(id=id,
                                     data=data,
                                     label=label,
                                     distance=distance)

        neighbor = Neighbor(id=np.int32(1),
                            data=np.asarray([0]),
                            distance=np.float32(.234),
                            label=np.int64(0))

        assert(neighbor == response_neighbor)

    @patch.object(Indexer, 'get_info', return_value=(1234, 10))
    def test_info(self, Indexer):
        """ Test case that asserts that the API correctly returns
            information about the data stored by the indexer
        """
        # Query the API for information about the indexer
        response = client.get("/info")
        response_json = response.json()

        assert(response_json["number_embeddings"] == 1234)
        assert(response_json["embedding_size"] == 10)

    @patch.object(Indexer, 'get_metrics', return_value=(12000, 0.00231))
    def test_metrics(self, Indexer):
        """ Test case that asserts that the API correctly returns
            indexer performance metrics
        """
        # Query the API for indexer performance metrics
        response = client.get("/metrics")
        response_json = response.json()

        assert(response_json["number_lookups"] == 12000)
        assert(response_json["average_time"] == 0.00231)

    @patch.object(Indexer, 'add', return_value=[0, 1, 2, 3])
    def test_add(self, Indexer):
        """ Test case that asserts that the API correctly adds
            items to the indexer
        """
        # Add items to the indexer
        response = client.post(
            "/add",
            json={
                "examples": [[0], [1], [2], [3]],
                "labels": [0, 1, 2, 3]
            }
        )
        response_json = response.json()
        response_ids = np.asarray(response_json)

        assert((response_ids == np.asarray([0, 1, 2, 3])).all())

    @patch.object(Indexer, 'remove', return_value=None)
    def test_remove(self, Indexer):
        """ Test case that asserts that the API correctly removes
            items to the indexer
        """
        # Delete the first item in the indexer
        response = client.delete("/delete/0")

        assert(response.status_code == 200)
