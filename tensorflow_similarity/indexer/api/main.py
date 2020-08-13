from typing import Optional, List, Any
from fastapi import FastAPI, Query
from pydantic import BaseModel
from tensorflow_similarity.indexer.indexer import Indexer
import os
import json
import numpy as np

app = FastAPI()

bundle_path = os.path.abspath(os.path.join(__file__, '../../', 'bundle'))
indexer = Indexer.load(bundle_path)
indexer.build()


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
            
            response.append({
                "id": id, 
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

            response.append({
                "id": id, 
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
    ids = indexer.add(np.asarray(item.examples),
                      item.labels, 
                      item.original_examples)

    return ids


@app.delete("/delete/{id}")
def delete(id: int):
    """ Delete an item from the indexer
    """
    indexer.remove(id)
    return {id}
