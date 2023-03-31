"""The module to handle FAISS search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from termcolor import cprint

from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.types import FloatTensor

from .search import Search


class FaissSearch(Search):
    """This class implements the Faiss ANN interface.

    It implements the Search interface.
    """

    def __init__(
        self,
        distance: Distance | str,
        dim: int,
        verbose: int = 0,
        name: str | None = None,
        algo="ivfpq",
        m=8,
        nbits=8,
        nlist=1024,
        nprobe=1,
        normalize=True,
        **kw_args,
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
        super().__init__(distance=distance, dim=dim, verbose=verbose, name=name)
        self.algo = algo
        self.m = m  # number of bits per subquantizer
        self.nbits = nbits
        self.nlist = nlist
        self.nprobe = nprobe
        self.normalize = normalize
        self.built = False

        if verbose:
            t_msg = [
                "\n|-Initialize NMSLib Index",
                f"|  - algo:         {self.algo}",
                f"|  - m:            {self.m}",
                f"|  - nbits:        {self.nbits}",
                f"|  - nlist:        {self.nlist}",
                f"|  - nprobe:       {self.nprobe}",
                f"|  - normalize:    {self.normalize}",
            ]
            cprint("\n".join(t_msg) + "\n", "green")
        self.reset()

    def reset(self):
        if self.algo == "ivfpq":
            assert self.dim % self.m == 0, f"dim={self.dim}, m={self.m}"
        if self.algo == "ivfpq":
            metric = faiss.METRIC_L2
            prefix = ""
            if self.distance == distance_canonicalizer("cosine"):
                prefix = "L2norm,"
                metric = faiss.METRIC_INNER_PRODUCT
                # this distance requires both the input and query vectors to be normalized
            ivf_string = f"IVF{self.nlist},"
            pq_string = f"PQ{self.m}x{self.nbits}"
            factory_string = prefix + ivf_string + pq_string
            self.index = faiss.index_factory(self.dim, factory_string, metric)
            # quantizer = faiss.IndexFlatIP(
            #     dim
            # )  # we keep the same L2 distance flat index
            # self.index = faiss.IndexIVFPQ(
            #     quantizer, dim, nlist, m, nbits, metric=faiss.METRIC_INNER_PRODUCT
            # )
            # else:
            #   quantizer = faiss.IndexFlatL2(
            #       dim
            #   )  # we keep the same L2 distance flat index
            #   self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            self.index.nprobe = self.nprobe  # set how many of nearest cells to search
        elif self.algo == "flat":
            if self.distance == distance_canonicalizer("cosine"):
                # this is exact match using cosine/dot-product Distance
                self.index = faiss.IndexFlatIP(self.dim)
            elif self.distance == distance_canonicalizer("l2"):
                # this is exact match using L2 distance
                self.index = faiss.IndexFlatL2(self.dim)
            else:
                raise ValueError(f"distance {self.distance} not supported")

    def is_built(self):
        return self.algo == "flat" or self.index.is_trained

    def build_index(self, samples, normalize=True, **kwargss):
        if self.algo == "ivfpq":
            if normalize:
                faiss.normalize_L2(samples)
            self.index.train(samples)  # we must train the index to cluster into cells
            self.built = True

    def batch_lookup(
        self, embeddings: FloatTensor, k: int = 5, normalize: bool = True
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Find embeddings K nearest neighboors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """

        if normalize:
            faiss.normalize_L2(embeddings)
        sims, indices = self.index.search(embeddings, k)
        return indices, sims

    def lookup(self, embedding: FloatTensor, k: int = 5, normalize: bool = True) -> tuple[list[int], list[float]]:
        """Find embedding K nearest neighboors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """
        int_embedding = np.array([embedding], dtype=np.float32)
        if normalize:
            faiss.normalize_L2(int_embedding)
        sims, indices = self.index.search(int_embedding, k)
        return indices[0], sims[0]

    def add(self, embedding: FloatTensor, idx: int, verbose: int = 1, normalize: bool = True, **kwargs):
        """Add a single embedding to the search index.

        Args:
            embedding: The embedding to index as computed by the similarity model.
            idx: Embedding id as in the index table. Returned with the embedding to
              allow to lookup the data associated with a given embedding.
        """
        int_embedding = np.array([embedding], dtype=np.float32)
        if normalize:
            faiss.normalize_L2(int_embedding)
        if self.algo != "flat":
            self.index.add_with_ids(int_embedding)
        else:
            self.index.add(int_embedding)

    def batch_add(
        self,
        embeddings: FloatTensor,
        idxs: Sequence[int],
        verbose: int = 1,
        normalize: bool = True,
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
        if normalize:
            faiss.normalize_L2(embeddings)
        if build and not self.is_built():
            print("building Faiss index")
            self.build_index(samples=embeddings, normalize=normalize)
        if self.algo != "flat":
            # flat does not accept indexes as parameters and assumes incremental
            # indexes
            self.index.add_with_ids(embeddings, np.array(idxs))
        else:
            self.index.add(embeddings)

    def save(self, path: str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """
        chunk = faiss.serialize_index(self.index)
        np.save(self.__make_fname(path), chunk)

    def __make_fname(self, path):
        return str(Path(path) / "faiss_index.npy")

    def load(self, path: str):
        """load index on disk

        Args:
            path: where to store the data
        """
        self.index = faiss.deserialize_index(np.load(self.__make_fname(path)))  # identical to index

    def get_config(self) -> dict[str, Any]:
        """Contains the search configuration.

        Returns:
            A Python dict containing the configuration of the search obj.
        """
        config = {
            "distance": self.distance.name,
            "dim": self.dim,
            "algo": self.algo,
            "m": self.m,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "normalize": self.normalize,
            "verbose": self.verbose,
            "name": self.name,
            "canonical_name": self.__class__.__name__,
        }
        base_config = super().get_config()
        return {**base_config, **config}
