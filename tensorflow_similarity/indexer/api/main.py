from typing import Optional, List
from fastapi import FastAPI, Query
from pydantic import BaseModel
from tensorflow_similarity.indexer.indexer import Indexer
import os
import json
import numpy as np

app = FastAPI()

with open(os.path.abspath("../config.json"), 'r') as config_file:
        config = json.load(config_file)
indexer_config = config["indexer"]
indexer = Indexer(dataset_examples_path=os.path.join("../", indexer_config.get("dataset")), 
                  dataset_labels_path=os.path.join("../", indexer_config.get("dataset_labels")), 
                      model_path=os.path.join("../", indexer_config.get("model")), 
                      dataset_original_path=os.path.join("../", indexer_config.get("original")),
                      space=indexer_config.get("space", "cosinesimil"))
indexer.build(verbose=indexer_config.get("verbose", 1))

class Neighbor(BaseModel):
    id: int
    data: List[bytes]
    distance: float
    label: int

class Item(BaseModel):
    num_neighbors: int
    embeddings: List[List[float]]

@app.post("/lookupEmbeddings")
def lookup_embeddings(item: Item):
    neighbors = indexer.find(items=item.embeddings,
                             num_neighbors=item.num_neighbors, 
                             is_embedding=True)
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


@app.post("/lookup", response_model=Neighbor)
def lookup(item: Item):
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