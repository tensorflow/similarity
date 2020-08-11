from typing import Optional, List, Any
from fastapi import FastAPI, Query
from pydantic import BaseModel
from tensorflow_similarity.indexer.indexer import Indexer
import os
import json
import numpy as np

app = FastAPI()

with open(os.path.abspath("../config.json"), 'r') as config_file:
        config = json.load(config_file)

indexer = Indexer.load("../bundle")
indexer.build()

class Neighbor(BaseModel):
    id: int
    data: List[Any]
    distance: float
    label: int

class Item(BaseModel):
    examples: List[List[float]]
    labels: List[int]
    original_examples: Optional[List[List[Any]]] = None

class LookupItem(BaseModel):
    num_neighbors: int
    embeddings: List[List[float]]

@app.post("/lookupEmbeddings")
def lookup_embeddings(item: LookupItem):
    neighbors = indexer.find(items=item.embeddings,
                             num_neighbors=item.num_neighbors, 
                             is_embedding=True)
    response = []
    for neighbor_list in neighbors:
        for neighbor_item in neighbor_list:
            id = np.asscalar(neighbor_item.id)
            data = np.asscalar(neighbor_item.data)
            distance = np.asscalar(neighbor_item.distance)
            label = np.asscalar(neighbor_item.label)

            response.append({"id": id, 
                            "data": data, 
                            "distance": distance, 
                            "label": label})

    return response


@app.post("/lookup", response_model=Neighbor)
def lookup(item: LookupItem):
    neighbors = indexer.find(items=item.embeddings,
                             num_neighbors=item.num_neighbors, 
                             is_embedding=False)
    response = []
    for neighbor_item in neighbors[0]:
        id = np.asscalar(neighbor_item.id)
        data = np.asscalar(neighbor_item.data)
        distance = np.asscalar(neighbor_item.distance)
        label = np.asscalar(neighbor_item.label)

        response.append({"id": id, 
                         "data": data, 
                         "distance": distance, 
                         "label": label})
    return response

@app.get("/info")
def info():
    num_embeddings = indexer.num_embeddings
    serving_directory = None
    embedding_size = indexer.embedding_size

    return {"number_embeddings": num_embeddings, 
            "serving_directory": serving_directory,
            "embedding_size": embedding_size}


@app.get("/metrics")
def metrics():
    num_lookups = indexer.num_lookups
    if num_lookups > 0:
        avg_query_time = indexer.lookup_time / num_lookups
    else:
        avg_query_time = 0

    return {"number_lookups": num_lookups,
            "average_time": avg_query_time}

@app.post("/add")
def add(item: Item):
    ids = indexer.add(np.asarray(item.examples), item.labels, item.original_examples)
    return ids
