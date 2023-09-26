"""The module to handle FAISS search."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from termcolor import cprint

from ..distances import (
    CosineDistance,
    EuclideanDistance,
    InnerProductSimilarity,
    ManhattanDistance,
    SNRDistance,
    SquaredEuclideanDistance,
)

if TYPE_CHECKING:
    from ..distances import Distance
    from collections.abc import Sequence
    from ..types import FloatTensor

from .search import Search


class FaissSearch(Search):
    """This class implements the Faiss ANN interface.

    It implements the Search interface.
    """

    def __init__(
        self,
        distance: Distance | str,
        dim: int,
        algo: str = "ivfpq",
        m: int = 8,
        nbits: int = 8,
        nlist: int = 1024,
        nprobe: int = 1,
        name: str = "faiss",
        verbose: int = 0,
        **kwargs,
    ):
        """Initiate FAISS indexer

        Args:
          d: number of dimensions
          m: number of centroid IDs in final compressed vectors. d must be divisible
            by m
          nbits: number of bits in each centroid
          nlist: how many Voronoi cells (must be greater than or equal to 2**nbits)
          nprobe: how many of the nearest cells to include in search
        """
        super().__init__(distance=distance, dim=dim, name=name, verbose=verbose)
        self.algo = algo
        self.m = m  # number of bits per subquantizer
        self.nbits = nbits
        self.nlist = nlist
        self.nprobe = nprobe

        self.reset()

    def add(self, embedding: FloatTensor | np.ndarray, idx: int, verbose: int = 1, **kwargs):
        """Add a single embedding to the search index.

        Args:
            embedding: The embedding to index as computed by
            the similarity model.

            idx: Embedding id as in the index table.
            Returned with the embedding to allow to lookup
            the data associated with a given embedding.

            verbose: Be verbose. Defaults to 1.
        """
        int_embedding = np.array([embedding], dtype=np.float32)
        if self.algo != "flat":
            self._index.add_with_ids(int_embedding)
        else:
            self._index.add(int_embedding)

    def batch_add(
        self,
        embeddings: FloatTensor | np.ndarray,
        idxs: Sequence[int],
        verbose: int = 1,
        build: bool = True,
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
        embeddings = np.asanyarray(embeddings, dtype=np.float32)
        if build and not self.is_built():
            print("building Faiss index")
            self.train_index(samples=embeddings)
        if self.algo != "flat":
            # flat does not accept indexes as parameters and assumes incremental
            # indexes
            self._index.add_with_ids(embeddings, np.asanyarray(idxs, dtype=np.int64))
        else:
            self._index.add(embeddings)

    def lookup(self, embedding: FloatTensor | np.ndarray, k: int = 5) -> tuple[list[int], list[float]]:
        """Find embedding K nearest neighbors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighbors embedding to lookup. Defaults to 5.
        """
        dists, idxs = self._index.search(
            np.array([embedding], dtype=np.float32),
            k,
        )
        # Filter out invalid indexes
        idxs = [i for i in idxs[0] if i != -1]
        dists = dists[0][: len(idxs)]
        if isinstance(self.distance, CosineDistance):
            # Faiss returns cosine similarity, but we want cosine distance
            dists = [1 - sim for sim in dists]
        return idxs, dists

    def batch_lookup(
        self, embeddings: FloatTensor | np.ndarray, k: int = 5
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Find embeddings K nearest neighbors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighbors embedding to lookup. Defaults to 5.
        """

        batch_idxs = []
        batch_distances = []

        embeddings = np.asanyarray(embeddings, dtype=np.float32)
        dists, idxs = self._index.search(embeddings, k)
        for d, ix in zip(dists, idxs):
            # Filter out invalid indexes
            batch_idxs.append([i for i in ix if i != -1])
            batch_distances.append(d[: len(batch_idxs[-1])])
            if isinstance(self.distance, CosineDistance):
                # Faiss returns cosine similarity, but we want cosine distance
                batch_distances[-1] = [1 - sim for sim in batch_distances[-1]]

        return batch_idxs, batch_distances

    def save(self, path: Path | str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """
        faiss = self._try_import_faiss()

        chunk = faiss.serialize_index(self._index)
        np.save(self._make_fname(path), chunk)

    def load(self, path: Path | str):
        """load index on disk

        Args:
            path: where to store the data
        """
        faiss = self._try_import_faiss()

        self._index = faiss.deserialize_index(np.load(self._make_fname(path)))  # identical to index

    def reset(self):
        faiss = self._try_import_faiss()
        self.built: bool = False
        if self.algo == "ivfpq":
            assert self.dim % self.m == 0, f"dim={self.dim}, m={self.m}"
            # this distance expects both the input and query vectors to be normalized
            if isinstance(self.distance, CosineDistance) or isinstance(self.distance, InnerProductSimilarity):
                prefix = "L2norm,"
                metric = faiss.METRIC_INNER_PRODUCT
            elif isinstance(self.distance, ManhattanDistance):
                prefix = ""
                metric = faiss.METRIC_L1
            else:
                prefix = ""
                metric = faiss.METRIC_L2
            ivf_string = f"IVF{self.nlist},"
            pq_string = f"PQ{self.m}x{self.nbits}"
            factory_string = prefix + ivf_string + pq_string
            self._index = faiss.index_factory(self.dim, factory_string, metric)
            # quantizer = faiss.IndexFlatIP(
            #     dim
            # )  # we keep the same L2 distance flat index
            # self._index = faiss.IndexIVFPQ(
            #     quantizer, dim, nlist, m, nbits, metric=faiss.METRIC_INNER_PRODUCT
            # )
            # else:
            #   quantizer = faiss.IndexFlatL2(
            #       dim
            #   )  # we keep the same L2 distance flat index
            #   self._index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            self._index.nprobe = self.nprobe  # set how many of nearest cells to search
        elif self.algo == "flat":
            if isinstance(self.distance, CosineDistance) or isinstance(self.distance, InnerProductSimilarity):
                # this is exact match using cosine/dot-product Distance
                self._index = faiss.IndexFlatIP(self.dim)
            elif isinstance(self.distance, ManhattanDistance):
                # this is exact match using L1 distance
                self._index = faiss.IndexFlat(self.dim, metric=faiss.METRIC_L1)
            elif (
                isinstance(self.distance, EuclideanDistance)
                or isinstance(self.distane, SquaredEuclideanDistance)  # noqa: W503
                or isinstance(self.distane, SNRDistance)  # noqa: W503
            ):
                # this is exact match using L2 distance
                self._index = faiss.IndexFlatL2(self.dim)
            else:
                raise ValueError(f"distance {self.distance} not supported")

        if self.verbose:
            t_msg = [
                "\n|-Initialize Faiss Index",
                f"|  - distance:       {self.distance}",
                f"|  - dim:            {self.dim}",
                f"|  - algo:         {self.algo}",
                f"|  - m:            {self.m}",
                f"|  - nbits:        {self.nbits}",
                f"|  - nlist:        {self.nlist}",
                f"|  - nprobe:       {self.nprobe}",
            ]
            cprint("\n".join(t_msg) + "\n", "green")

    def get_config(self) -> dict[str, Any]:
        """Contains the search configuration.

        Returns:
            A Python dict containing the configuration of the search obj.
        """
        config = super().get_config()
        config.update(
            {
                "algo": self.algo,
                "m": self.m,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
            }
        )
        return config

    def is_built(self):
        return self.algo == "flat" or self._index.is_trained

    def train_index(self, samples, **kwargss):
        if self.algo == "ivfpq":
            samples = np.asanyarray(samples, dtype=np.float32)
            self._index.train(samples)  # we must train the index to cluster into cells
            self.built = True

    def _make_fname(self, path):
        return Path(path) / "faiss_index.npy"

    def _try_import_faiss(self):
        try:
            import faiss
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "faiss is not installed. Please install it with `pip install tensorflow_similarity[faiss]`"
            ) from e
        return faiss
