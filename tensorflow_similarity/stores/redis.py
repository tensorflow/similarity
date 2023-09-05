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

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence
    from tensorflow_similarity.types import FloatTensor, PandasDataFrame, Tensor

from .store import Store


class RedisStore(Store):
    """Efficient Redis dataset store"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        name: str = "redis",
        verbose: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(name=name, verbose=verbose)
        # Currently does not support authentication
        self.host = host
        self.port = port
        self.db = db
        self._connect()

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
        idx = self.size()
        self._conn.set(str(idx), pickle.dumps((embedding, label, data)))
        self._conn.incr("num_items")

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
        for i, embedding in enumerate(embeddings):
            label = None if labels is None else labels[i]
            rec_data = None if data is None else data[i]
            idx = self.add(embedding, label, rec_data)
            idxs.append(idx)

        return idxs

    def get(self, idx: int) -> tuple[FloatTensor, int | None, Tensor | None]:
        """Get an embedding record from the key value store.

        Args:
            idx: Id of the record to fetch.

        Returns:
            record associated with the requested id.
        """

        ret_bytes: bytes = self._conn.get(str(idx))
        ret: tuple = pickle.loads(ret_bytes)
        return (ret[0], ret[1], ret[2])

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
        size: str | None = self._conn.get("num_items")
        return int(size) if size is not None else 0

    def save(self, path: Path | str, compression: bool = True) -> None:
        """Serializes index on disk.

        Args:
            path: where to store the data.
            compression: Compress index data. Defaults to True.
        """
        # Writing to a buffer to avoid read error in np.savez when using GFile.
        # See: https://github.com/tensorflow/tensorflow/issues/32090
        self._save_config(path)

    def load(self, path: Path | str) -> int:
        """load index on disk

        Args:
            path: which directory to use to store the index data.

        Returns:
           Number of records reloaded.
        """
        self._load_config(path)
        return self.size()

    def get_config(self):
        config = super().get_config()
        config.update({"host": self.host, "port": self.port, "db": self.db})
        return config

    def to_data_frame(self, num_records: int = 0) -> PandasDataFrame:
        """Export data as a Pandas dataframe.

        Args:
            num_records: Number of records to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            Empty DataFrame
        """
        if num_records:
            idxs = list(range(num_records))
            embeddings, labels, data = self.batch_get(idxs)
            records: dict[str, list[Any]] = {
                "embedding": embeddings,
                "label": labels,
                "data": data,
            }
        else:
            records = {
                "embedding": [],
                "label": [],
                "data": [],
            }

        # forcing type from Any to PandasFrame
        df: PandasDataFrame = pd.DataFrame.from_dict(records)
        return df

    def reset(self):
        self._conn.flushdb()

    def _make_config_file_path(self, path):
        return Path(path) / "config.json"

    def _save_config(self, path):
        with open(self._make_config_file_path(path), "wt") as f:
            json.dump(self.get_config(), f)

    def _set_config(self, host, port, db, **kw_args):
        self.host = host
        self.port = port
        self.db = db

    def _connect(self):
        import redis

        self._conn = redis.Redis(host=self.host, port=self.port, db=self.db)

    def _load_config(self, path):
        with open(self._make_config_file_path(path), "rt") as f:
            self._set_config(**json.load(f))
        self._connect()
