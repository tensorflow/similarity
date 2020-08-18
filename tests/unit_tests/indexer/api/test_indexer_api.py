from fastapi.testclient import TestClient
from tensorflow_similarity.indexer.api.main import app
import tensorflow_similarity.indexer.api.main as api
import numpy as np
from tensorflow_similarity.indexer.indexer import Indexer
import unittest
from mock import patch
import collections
import json
import uuid


client = TestClient(app)

Neighbor = collections.namedtuple("Neighbor",
                                  ["id",
                                   "data",
                                   "distance",
                                   "label"])


class IndexerTestCase(unittest.TestCase):

    def generate_mock():
        """ Generate a neighbor for mocking
        """
        neighbor_id = np.int32(1)
        neighbor_data = np.asarray([0])
        neighbor_distance = np.float32(.234)
        neighbor_label = np.int64(0)
        neighbor_mock = Neighbor(id=neighbor_id,
                                 data=neighbor_data,
                                 distance=neighbor_distance,
                                 label=neighbor_label)

        return [[neighbor_mock, neighbor_mock]]


    def convert_response_to_neighbor(self, response):
        """ Convert a response to a list neighbor namedtuple
        """
        neighbors = []
        response_neighbors = []

        for response_list in response.json():
            for response_item in response_list:
                id = response_item['id']
                label = response_item['label']
                data = response_item['data']
                distance = response_item['distance']

                response_neighbor = Neighbor(id=id,
                                            data=data,
                                            label=label,
                                            distance=distance)

                neighbor = Neighbor(id=response_neighbor.id,
                                    data=np.asarray([0]),
                                    distance=np.float32(0.234),
                                    label=np.int64(0))

                response_neighbors.append(response_neighbor)
                neighbors.append(neighbor)

        return neighbors, response_neighbors


    @patch.object(Indexer, 'find', return_value=generate_mock())
    def test_lookup_embeddings(self, Indexer):
        """ Test case that asserts that the API correctly performs a lookup
            of the nearest neighbors for a list of embeddings
        """
        # Generate embeddings
        embeddings = np.random.uniform(low=-1.0, high=0.99, size=(2, 4))
        embedding_list = embeddings.tolist()
        num_neighbors = 2

        # Query the API for the nearest neighbors of the embeddings
        response = client.post(
            "/lookupEmbeddings",
            json={"num_neighbors": num_neighbors, "data": embedding_list}
        )

        # Convert the response to a list of neighbor namedtuples
        neighbors, response_neighbors = self.convert_response_to_neighbor(response)
        neighbors = np.asarray(neighbors)
        response_neighbors = np.asarray(response_neighbors)

        assert((neighbors == response_neighbors).all())


    @patch.object(Indexer, 'find', return_value=generate_mock())
    def test_lookup(self, Indexer):
        """ Test case that asserts that the API correctly performs a lookup
            of the nearest neighbors for a list of examples
        """
        # Generate embeddings
        embeddings = np.random.randint(1000, size=(2, 28))
        embedding_list = embeddings.tolist()
        num_neighbors = 2

        # Query the API for the nearest neighbors of the embeddings
        response = client.post(
            "/lookup",
            json={"num_neighbors": num_neighbors, "data": embedding_list}
        )

        # Convert the response to a list of neighbor namedtuples
        neighbors, response_neighbors = self.convert_response_to_neighbor(response)
        neighbors = np.asarray(neighbors)
        response_neighbors = np.asarray(response_neighbors)

        assert((neighbors == response_neighbors).all())


    @patch.object(Indexer, 'get_info', return_value={"num_embeddings": 1234, 
                                                     "embedding_size": 10})
    def test_info(self, Indexer):
        """ Test case that asserts that the API correctly returns
            information about the data stored by the indexer
        """
        # Query the API for information about the indexer
        response = client.get("/info")
        response_json = response.json()

        assert(response_json["num_embeddings"] == 1234)
        assert(response_json["embedding_size"] == 10)


    @patch.object(Indexer, 'get_metrics', return_value={"num_lookups": 12000, 
                                                        "avg_query_time": 0.00231})
    def test_metrics(self, Indexer):
        """ Test case that asserts that the API correctly returns
            indexer performance metrics
        """
        # Query the API for indexer performance metrics
        response = client.get("/metrics")
        response_json = response.json()

        assert(response_json["num_lookups"] == 12000)
        assert(response_json["avg_query_time"] == 0.00231)


    @patch.object(Indexer, 'add', return_value=[0, 1, 2, 3])
    def test_add(self, Indexer):
        """ Test case that asserts that the API correctly adds
            items to the indexer
        """
        uuid_mock = patch.object(uuid, 'uuid1', return_value="1234")

        # Add items to the indexer
        with uuid_mock:
            response = client.post(
                "/add",
                json={
                    "examples": [[0], [1], [2], [3]],
                    "labels": [0, 1, 2, 3]
                }
            )
            response_json = response.json()
            response_ids = np.asarray(response_json)

            assert((response_ids == np.asarray(["1234"] * 4)).all())


    @patch.object(Indexer, 'remove', return_value=[0])
    def test_remove(self, Indexer):
        """ Test case that asserts that the API correctly removes
            items from the indexer
        """
        # Generate mock for uuid_id map
        uuids = [uuid.UUID('3dc13b90-e0ae-11ea-8b23-00163ee9c8c6'), 
                 uuid.UUID('3dc13b90-e0ae-11ea-8b23-00163eb9c8c6')]
        ids = [0, 1]
        uuid_id_map = {uuid_str: id for uuid_str, id in zip(uuids, ids)}

        api_mock = patch.object(api, 'uuid_id_map', uuid_id_map)

        # Delete the item from the indexer
        with api_mock:
            request_data = json.dumps([str(uuids[0])])
            response = client.delete("/delete", data=request_data)

            assert(uuids[0] not in api.uuid_id_map)
            assert(api.uuid_id_map[uuids[1]] == ids[1] - 1)
            assert(response.status_code == 200)
