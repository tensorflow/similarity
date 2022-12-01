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

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, PandasDataFrame, Tensor

from .store import Store


class MemoryStore(Store):
    """Efficient in-memory dataset store"""

    def __init__(self) -> None:
        # We are using a native python array in memory for its row speed.
        # Serialization / export relies on Arrow.
        self.labels: list[int | None] = []
        self.embeddings: list[FloatTensor] = []
        self.data: list[Tensor | None] = []
        self.num_items: int = 0
        pass

    def add(
        self,
        embedding: FloatTensor,
        label: int | None = None,
        data: Tensor | None = None,
    ) -> int:
        """Add an Embedding record to the key value store.

        Args:
            embedding: Embedding predicted by the model.

            label: Class numerical id. Defaults to None.

            data: Data associated with the embedding. Defaults to None.

        Returns:
            Associated record id.
        """
        idx = self.num_items
        self.labels.append(label)
        self.embeddings.append(embedding)
        self.data.append(data)
        self.num_items += 1
        return idx

    def batch_add(
        self,
        embeddings: Sequence[FloatTensor],
        labels: Sequence[int] | None = None,
        data: Sequence[Tensor] | None = None,
    ) -> list[int]:
        """Add a set of embedding records to the key value store.

        Args:
            embeddings: Embeddings predicted by the model.

            labels: Class numerical ids. Defaults to None.

            data: Data associated with the embeddings. Defaults to None.

        See:
            add() for what a record contains.

        Returns:
            List of associated record id.
        """
        idxs: list[int] = []
        for idx, embedding in enumerate(embeddings):
            label = None if labels is None else labels[idx]
            rec_data = None if data is None else data[idx]
            idxs.append(self.add(embedding, label, rec_data))
        return idxs

    def get(self, idx: int) -> tuple[FloatTensor, int | None, Tensor | None]:
        """Get an embedding record from the key value store.

        Args:
            idx: Id of the record to fetch.

        Returns:
            record associated with the requested id.
        """

        return self.embeddings[idx], self.labels[idx], self.data[idx]

    def batch_get(self, idxs: Sequence[int]) -> tuple[list[FloatTensor], list[int | None], list[Tensor | None]]:
        """Get embedding records from the key value store.

        Args:
            idxs: ids of the records to fetch.

        Returns:
            List of records associated with the requested ids.
        """
        embeddings = []
        labels = []
        data = []
        for idx in idxs:
            e, l, d = self.get(idx)
            embeddings.append(e)
            labels.append(l)
            data.append(d)
        return embeddings, labels, data

    def size(self) -> int:
        "Number of record in the key value store."
        return self.num_items

    def save(self, path: str, compression: bool = True) -> None:
        """Serializes index on disk.

        Args:
            path: where to store the data.
            compression: Compress index data. Defaults to True.
        """
        # Writing to a buffer to avoid read error in np.savez when using GFile.
        # See: https://github.com/tensorflow/tensorflow/issues/32090
        io_buffer = io.BytesIO()
        if compression:
            np.savez_compressed(
                io_buffer,
                embeddings=self.embeddings,
                labels=np.array(self.labels),
                data=np.array(self.data),
            )
        else:
            np.savez(
                io_buffer,
                embeddings=self.embeddings,
                labels=np.array(self.labels),
                data=np.array(self.data),
            )

        with tf.io.gfile.GFile(self._make_fname(path), "wb+") as f:
            f.write(io_buffer.getvalue())

    def load(self, path: str) -> int:
        """load index on disk

        Args:
            path: which directory to use to store the index data.

        Returns:
           Number of records reloaded.
        """
        fname = self._make_fname(path, check_file_exit=True)
        with tf.io.gfile.GFile(fname, "rb") as gfp:
            data = np.load(gfp, allow_pickle=True)
        self.embeddings = list(data["embeddings"])
        self.labels = list(data["labels"])
        self.data = list(data["data"])
        self.num_items = len(self.embeddings)
        print("loaded %d records from %s" % (self.size(), path))
        return self.size()

    def _make_fname(self, path: str, check_file_exit: bool = False) -> str:
        p = Path(path)
        if not tf.io.gfile.exists(p):
            raise ValueError("Index path doesn't exist")
        fname = p / "index.npz"

        # only for loading
        if check_file_exit and not tf.io.gfile.exists(fname):
            raise ValueError("Index file not found")
        return str(fname)

    def to_data_frame(self, num_records: int = 0) -> PandasDataFrame:
        """Export data as a Pandas dataframe.

        Args:
            num_records: Number of records to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """

        if not num_records:
            num_records = self.num_items

        data = {
            "embeddings": self.embeddings[:num_records],
            "data": self.data[:num_records],
            "lables": self.labels[:num_records],
        }

        # forcing type from Any to PandasFrame
        df: PandasDataFrame = pd.DataFrame.from_dict(data)
        return df
