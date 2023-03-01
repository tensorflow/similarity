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

from collections.abc import Sequence

import json
import pandas as pd
import pickle
import redis

from .store import Store

from tensorflow_similarity.types import FloatTensor, PandasDataFrame, Tensor


class RedisStore(Store):
    """Efficient Redis dataset store"""

    def __init__(self, host="localhost", port=6379, db=0) -> None:
        # Currently does not support authentication
        self.host = host
        self.port = port
        self.db = db
        self.__connect()

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
        num_items = int(self.__conn.incr("num_items"))
        idx = num_items - 1
        self.__conn.set(idx, pickle.dumps((embedding, label, data)))

        return idx

    def get_num_items(self) -> int:
        return int(self.__conn.get("num_items")) or 0

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

        ret = pickle.loads(self.__conn.get(idx))
        return ret[0], ret[1], ret[2]

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
        return self.get_num_items()

    def __make_config_file_path(self, path):
        return path / "config.json"

    def __save_config(self, path):
        with open(self.__make_config_file_path(path), "wt") as f:
            json.dump(self.get_config(), f)

    def __set_config(self, host, port, db):
        self.host = host
        self.port = port
        self.db = db

    def __connect(self):
        self.__conn = redis.Redis(host=self.host, port=self.port, db=self.db)

    def __load_config(self, path):
        with open(self.__make_config_file_path(path), "rt") as f:
            self.__set_config(**json.load(f))
        self.__connect()

    def save(self, path: str, compression: bool = True) -> None:
        """Serializes index on disk.

        Args:
            path: where to store the data.
            compression: Compress index data. Defaults to True.
        """
        # Writing to a buffer to avoid read error in np.savez when using GFile.
        # See: https://github.com/tensorflow/tensorflow/issues/32090
        self.__save_config(path)

    def get_config(self):
        return {"host": self.host, "port": self.port, "db": self.db, "num_items": self.get_num_items()}

    def load(self, path: str) -> int:
        """load index on disk

        Args:
            path: which directory to use to store the index data.

        Returns:
           Number of records reloaded.
        """
        self.__load_config(path)
        return self.size()

    def to_data_frame(self, num_records: int = 0) -> PandasDataFrame:
        """Export data as a Pandas dataframe.

        Cached store does not fit in memory, therefore we do not implement this.

        Args:
            num_records: Number of records to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            Empty DataFrame
        """
        # forcing type from Any to PandasFrame
        df: PandasDataFrame = pd.DataFrame()
        return df