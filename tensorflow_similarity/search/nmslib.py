# Copyright 2021 The TensorFlow Authors
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
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tensorflow as tf
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
    from collections.abc import Mapping, Sequence
    from ..types import FloatTensor

from .search import Search


class NMSLibSearch(Search):
    """
    Efficiently find nearest embeddings by indexing known embeddings and make them searchable using the
    [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
    search library [NMSLIB](https://github.com/nmslib/nmslib).
    """

    def __init__(
        self,
        distance: Distance | str,
        dim: int,
        method: str = "hnsw",
        space_params: Mapping[str, Any] | None = None,
        data_type: int = 0,  # nmslib.DataType.DENSE_VECTOR
        dtype: int = 0,  # nmslib.DistType.FLOAT
        index_params: Mapping[str, Any] | None = None,
        query_params: Mapping[str, Any] | None = None,
        name: str = "nmslib",
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(distance=distance, dim=dim, name=name, verbose=verbose)
        nmslib = self._try_import_nmslib()

        # Disable the INFO logging from NMSLIB
        logging.getLogger("nmslib").setLevel(logging.WARNING)

        self.method = method
        self.data_type = nmslib.DataType(data_type) if isinstance(data_type, int) else data_type
        self.dtype = nmslib.DistType(dtype) if isinstance(dtype, int) else dtype
        self.space_params = space_params
        self.index_params = index_params
        self.query_params = query_params

        self.reset()

    def add(self, embedding: FloatTensor, idx: int, verbose: int = 1, build: bool = True, **kwargs):
        """Add an embedding to the index

        Args:
            embedding: The embedding to index as computed by
            the similarity model.

            idx: Embedding id as in the index table.
            Returned with the embedding to allow to lookup
            the data associated with a given embedding.

            verbose: Be verbose. Defaults to 1.

            build: Rebuild the index after the addition.
            Required to make the embedding searchable.
            Set to false to save time between successive addition.
            Defaults to True.

        """
        self._index.addDataPoint(idx, embedding)
        if build:
            self._build(verbose=verbose)
        else:
            self.built = False

    def batch_add(self, embeddings: FloatTensor, idxs: Sequence[int], verbose: int = 1, build: bool = True, **kwargs):
        """Add a batch of embeddings to the search index.

        Args:
            embeddings: List of embeddings to add to the index.

            idxs (int): Embedding ids as in the index table. Returned with
            the embeddings to allow to lookup the data associated
            with the returned embeddings.

            verbose: Be verbose. Defaults to 1.

            build: Rebuild the index after the addition. Required to
            make the embeddings searchable. Set to false to save
            time between successive addition. Defaults to True.
        """
        # !addDataPoint and addDataPointBAtch have inverted parameters
        if verbose:
            print("|-Adding embeddings to index.")
        self._index.addDataPointBatch(embeddings, idxs)

        if build:
            if verbose:
                print("|-Building index.")
            self._build(verbose=verbose)
        else:
            self.built = False

    def lookup(self, embedding: FloatTensor, k: int = 5) -> tuple[list[int], list[float]]:
        """Find embedding K nearest neighboors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """
        idxs: list[int] = []
        distances: list[float] = []
        idxs, distances = self._index.knnQuery(embedding, k=k)
        return idxs, distances

    def batch_lookup(self, embeddings: FloatTensor, k: int = 5) -> tuple[list[list[int]], list[list[float]]]:
        """Find embeddings K nearest neighboors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """
        batch_idxs = []
        batch_distances = []

        nn = self._index.knnQueryBatch(embeddings, k=k)
        for n in nn:
            batch_idxs.append(n[0])
            batch_distances.append(n[1])
        return batch_idxs, batch_distances

    def save(self, path: Path | str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            fname = self._make_fname(path)
            tmpidx = Path(tmpdirname) / Path(path).name
            self._index.saveIndex(str(tmpidx), save_data=True)

            # read search_index.bin tmp file and write to fname
            with tf.io.gfile.GFile(fname, "w+b") as gfp:
                with open(tmpidx, "rb") as fp:
                    gfp.write(fp.read())

            # read search_index.bin.dat tmp file and write to fname + .dat
            with tf.io.gfile.GFile(fname.with_suffix(".dat"), "w+b") as gfp:
                tmpdat = Path(tmpdirname) / Path(path).with_suffix(".dat").name
                with open(tmpdat, "rb") as fp:
                    gfp.write(fp.read())

    def load(self, path: Path | str):
        """load index on disk

        Args:
            path: where to store the data
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            fname = self._make_fname(path)
            tmpidx = Path(tmpdirname) / Path(path).name

            # read fname and write to tmpidx
            with tf.io.gfile.GFile(fname, "rb") as gfp:
                with open(tmpidx, "w+b") as fp:
                    fp.write(gfp.read())

            # read fname + .dat and write to tmpdat
            with tf.io.gfile.GFile(fname.with_suffix(".dat"), "rb") as gfp:
                tmpdat = Path(tmpdirname) / Path(path).with_suffix(".dat").name
                with open(tmpdat, "w+b") as fp:
                    fp.write(gfp.read())

            self._index.loadIndex(str(tmpidx), load_data=True)

    def reset(self):
        nmslib = self._try_import_nmslib()

        self.built: bool = False
        if isinstance(self.distance, CosineDistance) or isinstance(self.distance, InnerProductSimilarity):
            space = "cosinesimil"
        elif (
            isinstance(self.distance, EuclideanDistance)
            or isinstance(self.distane, SquaredEuclideanDistance)  # noqa: W503
            or isinstance(self.distane, SNRDistance)  # noqa: W503
        ):
            space = "l2"
        elif isinstance(self.distance, ManhattanDistance):
            space = "l1"
        else:
            raise ValueError("Unsupported metric space")

        self._index = nmslib.init(
            space=space,
            space_params=self.space_params,
            method=self.method,
            data_type=self.data_type,
            dtype=self.dtype,
        )
        self._index.createIndex(index_params=self.index_params)
        self._index.setQueryTimeParams(params=self.query_params)

        if self.verbose:
            t_msg = [
                "\n|-Initialize NMSLib Index",
                f"|  - distance:     {self.distance}",
                f"|  - dim:          {self.dim}",
                f"|  - space:        {space}",
                f"|  - method:       {self.method}",
                f"|  - data_type:    {self.data_type}",
                f"|  - dist_type:    {self.dtype}",
                f"|  - space_params: {self.space_params}",
                f"|  - index_params: {self.index_params}",
                f"|  - query_params: {self.query_params}",
            ]
            cprint("\n".join(t_msg) + "\n", "green")

    def _build(self, verbose=0):
        """Build the index this is need to take into account the new points"""
        show = True if verbose else False
        self._index.createIndex(index_params=self.index_params, print_progress=show)
        self.built = True

    def _make_fname(self, path):
        return Path(path) / "search_index.bin"

    def get_config(self) -> dict[str, Any]:
        """Contains the search configuration.

        Returns:
            A Python dict containing the configuration of the search obj.
        """
        config = super().get_config()
        config.update(
            {
                "method": self.method,
                "space_params": self.space_params,
                "data_type": int(self.data_type),
                "dtype": int(self.dtype),
                "index_params": self.index_params,
                "query_params": self.query_params,
            }
        )

        return config

    def _try_import_nmslib(self):
        try:
            import nmslib
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "nmslib is not installed. Please install it with `pip install tensorflow_similarity[nmslib]`"
            ) from e
        return nmslib
