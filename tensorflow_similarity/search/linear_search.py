"""The module to handle Linear search."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List

import numpy as np
import tensorflow as tf
from termcolor import cprint

from tensorflow_similarity.distances import Distance
from tensorflow_similarity.types import FloatTensor

from .search import Search

INITIAL_DB_SIZE = 10000
DB_SIZE_STEPS = 10000


class LinearSearch(Search):
    """This class implements the Linear Search interface.

    It implements the Search interface.
    """

    def __init__(self, distance: Distance | str, dim: int, verbose: int = 0, name: str | None = None, **kw_args):
        """Initiate Linear indexer.

        Args:
          d: number of dimensions
          m: number of centroid IDs in final compressed vectors. d must be divisible
            by m
          nbits: number of bits in each centroid
          nlist: how many Voronoi cells (must be greater than or equal to 2**nbits)
          nprobe: how many of the nearest cells to include in search
        """
        super().__init__(distance=distance, dim=dim, verbose=verbose, name=name)

        if verbose:
            t_msg = [
                "\n|-Initialize NMSLib Index",
                f"|  - distance:       {self.distance}",
                f"|  - dim:            {self.dim}",
                f"|  - verbose:        {self.verbose}",
                f"|  - name:           {self.name}",
            ]
            cprint("\n".join(t_msg) + "\n", "green")
        self.reset()

    def is_built(self):
        return True

    def needs_building(self):
        return False

    def batch_lookup(
        self, embeddings: FloatTensor, k: int = 5, normalize: bool = True
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Find embeddings K nearest neighboors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """

        if normalize:
            query = tf.math.l2_normalize(embeddings, axis=1)
        else:
            query = embeddings
        db_tensor = tf.convert_to_tensor(self.db)
        dists = self.distance(query, db_tensor)
        dists, id_idxs = tf.math.top_k(tf.math.negative(dists), k)
        dists = tf.math.negative(dists)
        id_idxs = id_idxs.numpy()
        ids_array = np.array(self.ids)
        return list(np.array([ids_array[x] for x in id_idxs])), list(dists)

    def lookup(self, embedding: FloatTensor, k: int = 5, normalize: bool = True) -> tuple[list[int], list[float]]:
        """Find embedding K nearest neighboors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """
        embeddings: FloatTensor = tf.convert_to_tensor([embedding], dtype=np.float32)
        idxs, dists = self.batch_lookup(embeddings, k=k, normalize=normalize)
        return idxs[0], dists[0]

    def add(self, embedding: FloatTensor, idx: int, verbose: int = 1, normalize: bool = True, **kwargs):
        """Add a single embedding to the search index.

        Args:
            embedding: The embedding to index as computed by the similarity model.
            idx: Embedding id as in the index table. Returned with the embedding to
              allow to lookup the data associated with a given embedding.
        """
        if normalize:
            embedding = tf.math.l2_normalize(np.array([embedding], dtype=tf.keras.backend.floatx()), axis=1)[0]
        self.ids.append(idx)
        self.db.append(embedding)

    def batch_add(
        self,
        embeddings: FloatTensor,
        idxs: Sequence[int],
        verbose: int = 1,
        normalize: bool = True,
        **kwargs,
    ):
        """Add a batch of embeddings to the search index.

        Args:
            embeddings: List of embeddings to add to the index.
            idxs (int): Embedding ids as in the index table. Returned with the
              embeddings to allow to lookup the data associated with the returned
              embeddings.
            verbose: Be verbose. Defaults to 1.
        """
        if normalize:
            embeddings = tf.math.l2_normalize(embeddings, axis=1)
        self.ids.extend(idxs)
        self.db.extend(embeddings)

    def __make_file_path(self, path):
        return Path(path) / "index.pickle"

    def save(self, path: str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """
        with open(self.__make_file_path(path), "wb") as f:
            pickle.dump((self.db, self.ids), f)
        self.__save_config(path)

    def load(self, path: str):
        """load index on disk

        Args:
            path: where to store the data
        """
        with open(self.__make_file_path(path), "rb") as f:
            data = pickle.load(f)
        self.db = data[0]
        self.ids = data[1]

    def reset(self):
        self.db: List[FloatTensor] = []
        self.ids: List[int] = []

    def __make_config_path(self, path):
        return Path(path) / "config.json"

    def __save_config(self, path):
        with open(self.__make_config_path(path), "wt") as f:
            json.dump(self.get_config(), f)

    def get_config(self) -> dict[str, Any]:
        """Contains the search configuration.

        Returns:
            A Python dict containing the configuration of the search obj.
        """
        config = {
            "distance": self.distance.name,
            "dim": self.dim,
        }

        base_config = super().get_config()
        return {**base_config, **config}
