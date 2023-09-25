"""The module to handle Linear search."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf
from termcolor import cprint

if TYPE_CHECKING:
    from collections.abc import Sequence
    from ..distances import Distance
    from ..types import FloatTensor

from .search import Search

INITIAL_DB_SIZE = 10000
DB_SIZE_STEPS = 10000


class LinearSearch(Search):
    """This class implements the Linear Search interface.

    It implements the Search interface.
    """

    def __init__(self, distance: Distance | str, dim: int, name: str = "linear", verbose: int = 0, **kwargs):
        """Initiate Linear indexer.

        Args:
            distance: the distance used to compute the distance between
            embeddings.

            dim: the size of the embeddings.

            verbose: be verbose.
        """
        super().__init__(distance=distance, dim=dim, name=name, verbose=verbose)

        self.reset()

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
        self._index.append(embedding)

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
        self._index.extend(embeddings)

    def lookup(self, embedding: FloatTensor, k: int = 5, normalize: bool = True) -> tuple[list[int], list[float]]:
        """Find embedding K nearest neighbors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighbors embedding to lookup. Defaults to 5.
        """
        embeddings: FloatTensor = tf.convert_to_tensor([embedding], dtype=np.float32)
        idxs, dists = self.batch_lookup(embeddings, k=k, normalize=normalize)
        return idxs[0], dists[0]

    def batch_lookup(
        self, embeddings: FloatTensor, k: int = 5, normalize: bool = True
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Find embeddings K nearest neighbors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighbors embedding to lookup. Defaults to 5.
        """

        if normalize:
            query = tf.math.l2_normalize(embeddings, axis=1)
        else:
            query = embeddings
        db_tensor = tf.convert_to_tensor(np.array(self._index), dtype=query.dtype)
        dists = self.distance(query, db_tensor)
        # Clip K in case the index is smaller than K
        k = min(k, tf.shape(dists)[1])
        # NOTE: kTop K takes the largest K elements, so we need to negate
        # the distances to get the top K smallest distances.
        # NOTE: dists and id_idxs will be a tensor of shape (batch_size, k),
        # with each row a top_k result set of k elements.
        dists, id_idxs = tf.math.top_k(tf.math.negative(dists), k)
        dists = tf.math.negative(dists)
        id_idxs = id_idxs.numpy()
        ids_array = np.array(self.ids)
        return list(np.array([ids_array[x] for x in id_idxs])), list(dists)

    def save(self, path: Path | str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """
        with open(self._make_file_path(path), "wb") as f:
            pickle.dump((self._index, self.ids), f)
        self._save_config(path)

    def load(self, path: Path | str):
        """load index on disk

        Args:
            path: where to store the data
        """
        with open(self._make_file_path(path), "rb") as f:
            data = pickle.load(f)
        self._index = data[0]
        self.ids = data[1]

    def reset(self):
        self._index: list[FloatTensor] = []
        self.ids: list[int] = []
        self.built = True

        if self.verbose:
            t_msg = [
                "\n|-Initialize Linear Index",
                f"|  - distance:       {self.distance}",
                f"|  - dim:            {self.dim}",
            ]
            cprint("\n".join(t_msg) + "\n", "green")

    def get_config(self) -> dict[str, Any]:
        """Contains the search configuration.

        Returns:
            A Python dict containing the configuration of the search obj.
        """
        config = super().get_config()
        return config

    def is_built(self):
        return self.built

    def _make_config_path(self, path):
        return Path(path) / "config.json"

    def _save_config(self, path):
        with open(self._make_config_path(path), "wt") as f:
            json.dump(self.get_config(), f)

    def _make_file_path(self, path):
        return Path(path) / "index.pickle"
