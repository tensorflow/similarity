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

from typing import Optional, List, Any
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow_similarity.indexer.indexer import Indexer
import os
import numpy as np
import uuid

app = FastAPI()

# Load a saved indexer
bundle_path = os.path.abspath(os.path.join(__file__, '../../', 'bundle'))
indexer = Indexer.load(bundle_path)

# Dictionary mapping an items data to its uuid, used to retrieve the uuid
# of an item in the indexer. The API returns UUIDs for items in the indexer
# in order to ensure consistent delete operations.
data_uuid_map = {}

# Dictionary mapping an items uuid to its index in the dataset
uuid_id_map = {}


class Item(BaseModel):
    """ Schema for request used to add items to the indexer

        attributes:
            examples (list(list(float))): A list of the items to be added
                                          to the index.
            labels (list(int)): A list of the labels corresponding to the
                                items.
            original_examples (list(list(any))): A list of the original data
                                                 points if different from
                                                 examples. Defaults to None.
    """
    examples: List[List[float]]
    labels: List[int]
    original_examples: Optional[List[List[Any]]] = None


class LookupItem(BaseModel):
    """ Schema for requests that find the nearest neighbors
        within a dataset.

        attributes:
            num_neighbors (int): Number of neighbors that should be returned.
            data (list(list(any))): The data for which a query of the most
                                    similar items should be performed. Either
                                    a list of embeddings or a list of examples.
    """
    num_neighbors: int
    data: List[List[Any]]


def covert_neighbors_to_json(neighbors):
    """ Convert a list of neighbor the a json serializable list of dictionary

        args:
            neighbors list(list(Neighbor)): List of nearest neighbor item lists
                                            sorted by distance for each item
                                            that the query was performed on.

        returns:
            reponse (json): A JSON serializable list of nearest neighbors.
    """
    response = []

    for neighbor_list in neighbors:
        response_list = []
        for neighbor_item in neighbor_list:
            # Convert neighbor item to a JSON serializable dictionary
            id = np.asscalar(neighbor_item.id)
            data = neighbor_item.data.tolist()
            distance = np.asscalar(neighbor_item.distance)
            label = np.asscalar(neighbor_item.label)

            # Convert data to a hashable key
            data_key = tuple(neighbor_item.data.flatten().tolist())

            if data_key not in data_uuid_map:
                # Generate UUID, map items data to the UUID and
                # map the UUID to the items index in the dataset
                generated_uuid = uuid.uuid1()
                data_uuid_map[data_key] = generated_uuid
                uuid_id_map[generated_uuid] = id

            item_uuid = data_uuid_map[data_key]

            response_list.append({
                "id": item_uuid,
                "data": data,
                "distance": distance,
                "label": label
            })

        response.append(response_list)

    return response


@app.post("/lookupEmbeddings")
def lookup_embeddings(items: LookupItem):
    """ Find the nearest neighbors within the target data set
        for the embeddings received in the request
    """
    neighbors = indexer.find(items=items.data,
                             num_neighbors=items.num_neighbors,
                             is_embedding=True)

    response = covert_neighbors_to_json(neighbors)

    return response


@app.post("/lookup")
def lookup(items: LookupItem):
    """ Find the nearest neighbors within the target data set
        for the datapoints received in the request
    """
    neighbors = indexer.find(items=items.data,
                             num_neighbors=items.num_neighbors,
                             is_embedding=False)

    response = covert_neighbors_to_json(neighbors)

    return response


@app.get("/info")
def info():
    """ Get information about the indexer
    """
    info = indexer.get_info()
    info["serving_directory"] = bundle_path

    return info


@app.get("/metrics")
def metrics():
    """ Get performance metrics from the indexer
    """
    metrics = indexer.get_metrics()

    return metrics


@app.post("/add", response_model=List[str])
def add(item: Item):
    """ Add item(s) to the index
    """
    # Add the items to the indexer
    ids = indexer.add(np.asarray(item.examples),
                      item.labels,
                      item.original_examples)
    uuids = []

    for id in ids:
        # Generate a UUID for item
        generated_uuid = uuid.uuid1()
        uuids.append(str(generated_uuid))

        # Get the data associated with the item in the dataset
        # and convert it to a hashable key
        data = indexer.dataset_original[id]
        data_key = tuple(data.flatten().tolist())

        # Map the data to the UUID
        data_uuid_map[data_key] = generated_uuid

        # Map the UUID to the items index in the dataset
        uuid_id_map[generated_uuid] = id

    return uuids


@app.delete("/delete")
def remove(uuids: List[str]):
    """ Delete item(s) from the indexer
    """
    indices_deleted = []

    for item_uuid in uuids:
        # Get the items index
        uuid_hex = uuid.UUID(item_uuid)
        id = uuid_id_map[uuid_hex]
        indices_deleted.append(id)

        # Get the data associated with the item in the dataset
        # and convert it to a hashable key
        data = indexer.dataset_original[id]
        data_key = tuple(data.flatten().tolist())

        # Delete item uuid from maps
        data_uuid_map.pop(data_key)
        uuid_id_map.pop(uuid_hex)

    # Remove the item(s) from the indexer
    indexer.remove(indices_deleted)

    # Update the indices of the items in the maps to account
    # for deleted indices
    for item_uuid, id in uuid_id_map.items():
        for deleted_id in indices_deleted:
            if id > deleted_id:
                id = id - 1
        uuid_id_map[item_uuid] = id

    return uuids
