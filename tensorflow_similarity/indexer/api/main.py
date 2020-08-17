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

bundle_path = os.path.abspath(os.path.join(__file__, '../../', 'bundle'))
indexer = Indexer.load(bundle_path)

data_uuid_map = {}
uuid_id_map = {}

class Item(BaseModel):
    examples: List[List[Any]]
    labels: List[int]
    original_examples: Optional[List[List[Any]]] = None


class LookupItem(BaseModel):
    num_neighbors: int
    embeddings: List[List[Any]]


@app.post("/lookupEmbeddings")
def lookup_embeddings(item: LookupItem):
    """ Find the nearest neighbors for the embeddings
        received in the request
    """
    neighbors = indexer.find(items=item.embeddings,
                             num_neighbors=item.num_neighbors,
                             is_embedding=True)
    response = []

    for neighbor_list in neighbors:
        for neighbor_item in neighbor_list:
            # Convert neighbor item to a JSON serializable dictionary
            # and append it to the response
            id = np.asscalar(neighbor_item.id)
            data = neighbor_item.data.tolist()
            distance = np.asscalar(neighbor_item.distance)
            label = np.asscalar(neighbor_item.label)
            
            data_key = tuple(neighbor_item.data.flatten().tolist())

            if not data_key in data_uuid_map:
                # Generate UUID, map items data to the UUID and 
                # map the UUID to the items index in the dataset
                generated_uuid = uuid.uuid1()
                data_uuid_map[data_key] = generated_uuid
                uuid_id_map[generated_uuid] = id

            item_uuid = data_uuid_map[data_key]

            response.append({
                "id": item_uuid,
                "data": data,
                "distance": distance,
                "label": label
            })

    return response


@app.post("/lookup")
def lookup(item: LookupItem):
    """ Find the nearest neighbors for the datapoints
        received in the request
    """
    neighbors = indexer.find(items=np.asarray(item.embeddings),
                             num_neighbors=item.num_neighbors,
                             is_embedding=False)
    response = []

    for neighbor_list in neighbors:
        for neighbor_item in neighbor_list:
            # Convert neighbor item to a JSON serializable dictionary
            # and append it to the response
            id = np.asscalar(neighbor_item.id)
            data = neighbor_item.data.tolist()
            distance = np.asscalar(neighbor_item.distance)
            label = np.asscalar(neighbor_item.label)

            # Convert data to a hashable key
            data_key = tuple(neighbor_item.data.flatten().tolist())

            if not data_key in data_uuid_map:
                # Generate UUID, map items data to the UUID and 
                # map the UUID to the items index in the dataset
                generated_uuid = uuid.uuid1()
                data_uuid_map[data_key] = generated_uuid
                uuid_id_map[generated_uuid] = id

            item_uuid = data_uuid_map[data_key]
            
            response.append({
                "id": item_uuid,
                "data": data,
                "distance": distance,
                "label": label
            })

    return response


@app.get("/info")
def info():
    """ Get information about the indexer
    """
    (num_embeddings, embedding_size) = indexer.get_info()

    return {
        "number_embeddings": num_embeddings,
        "serving_directory": bundle_path,
        "embedding_size": embedding_size
    }


@app.get("/metrics")
def metrics():
    """ Get performance metrics from the indexer
    """
    (num_lookups, avg_query_time) = indexer.get_metrics()

    return {
        "number_lookups": num_lookups,
        "average_time": avg_query_time
    }


@app.post("/add", response_model=List[int])
def add(item: Item):
    """ Add item(s) to the index
    """
    # Add the items to the indexer
    ids = indexer.add(np.asarray(item.examples),
                      item.labels,
                      item.original_examples)

    for id in ids:
        # Generate a UUID for item
        generated_uuid = uuid.uuid1()

        # Get the data associated with the item in the dataset
        # and convert it to a hashable key
        data = indexer.dataset_original[id]
        data_key = tuple(data.flatten().tolist())

        # Map the data to the UUID
        data_uuid_map[data_key] = generated_uuid

        # Map the UUID to the items index in the dataset
        uuid_id_map[generated_uuid] = id

    return ids


@app.delete("/delete")
def remove(ids: List[str]):
    """ Delete an item from the indexer
    """
    indices_deleted = []
    items_deleted = 0
    rebuild_index = False

    for item_uuid in ids:
        # Get the item from 
        uuid_hex = uuid.UUID(item_uuid)
        id = uuid_id_map[uuid_hex] - items_deleted
        data = indexer.dataset_original[id]
        indices_deleted.append(id)

        # Get the data associated with the item in the dataset
        # and convert it to a hashable key
        data = indexer.dataset_original[id]
        data_key = tuple(data.flatten().tolist())
        
        # Delete item uuid from maps
        data_uuid_map.pop(data_key)
        uuid_id_map.pop(uuid_hex)
        items_deleted = items_deleted + 1

        if items_deleted == len(ids):
            rebuild_index = True

        # Remove the item from the indexer
        indexer.remove(id, rebuild_index=rebuild_index)
        
    # Update the indices of the items in the maps to account 
    # for deleted indices
    for item_uuid, id in uuid_id_map.items():
        for deleted_id in indices_deleted:
            if id > deleted_id:
                id = id - 1
        uuid_id_map[item_uuid] = id

    return {tuple(ids)}
