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
import math
import pickle
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tensorflow_similarity.types import FloatTensor, PandasDataFrame, Tensor

from .store import Store


class CachedStore(Store):
    """Efficient cached dataset store"""

    def __init__(
        self,
        shard_size: int = 1000000,
        path: Path | str = ".",
        num_items: int = 0,
        name: str = "cached",
        verbose: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(name=name, verbose=verbose)
        # We are using a native python cached dictionary
        # db[id] = pickle((embedding, label, data))
        self.db: list[dict[str, bytes]] = []
        self.shard_size = shard_size
        self.num_items: int = num_items
        self.path: Path = Path(path)

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
        shard_no = self._get_shard_no(idx)
        if len(self.db) <= shard_no:
            self._add_new_shard()
        self.db[shard_no][str(idx)] = pickle.dumps((embedding, label, data))
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
        for i, embedding in enumerate(embeddings):
            idx = i + self.num_items
            label = None if labels is None else labels[i]
            rec_data = None if data is None else data[i]
            shard_no = self._get_shard_no(idx)
            if len(self.db) <= shard_no:
                self._add_new_shard()
            self.db[shard_no][str(idx)] = pickle.dumps((embedding, label, rec_data))
            idxs.append(idx)
        self.num_items += len(embeddings)

        return idxs

    def get(self, idx: int) -> tuple[FloatTensor, int | None, Tensor | None]:
        """Get an embedding record from the key value store.

        Args:
            idx: Id of the record to fetch.

        Returns:
            record associated with the requested id.
        """

        shard_no = self._get_shard_no(idx)
        embedding, label, data = pickle.loads(self.db[shard_no][str(idx)])
        return embedding, label, data

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

    def save(self, path: Path | str, compression: bool = True) -> None:
        """Serializes index on disk.

        Args:
            path: where to store the data.
            compression: Compress index data. Defaults to True.
        """
        # Writing to a buffer to avoid read error in np.savez when using GFile.
        # See: https://github.com/tensorflow/tensorflow/issues/32090
        self._close_all_shards()
        self._copy_shards(path)
        self._save_config(path)
        self._reopen_all_shards()

    def load(self, path: Path | str) -> int:
        """load index on disk

        Args:
            path: which directory to use to store the index data.

        Returns:
           Number of records reloaded.
        """
        self._load_config(path)
        num_shards = int(math.ceil(self.num_items / self.shard_size))
        self.path = Path(path)
        for i in range(num_shards):
            self._add_new_shard()
        return self.size()

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
        shard_no = len(self.db)
        self.db = []
        self.num_items = 0
        for i in range(shard_no):
            self._delete_shard(i)

    def get_config(self):
        config = super().get_config()
        config.update({"shard_size": self.shard_size, "num_items": self.num_items})
        return config

    def _get_shard_file_path(self, shard_no):
        return self.path / f"cache{shard_no}"

    def _make_new_shard(self, shard_no: int):
        import dbm.dumb

        return dbm.dumb.open(str(self._get_shard_file_path(shard_no)), "c")

    def _add_new_shard(self):
        shard_no = len(self.db)
        self.db.append(self._make_new_shard(shard_no))

    def _delete_shard(self, n):
        # TODO(ovallis): This is not actually deleting the file.
        self._get_shard_file_path(n)

    def _reopen_all_shards(self):
        for shard_no in range(len(self.db)):
            self.db[shard_no] = self._make_new_shard(shard_no)

    def _get_shard_no(self, idx: int) -> int:
        return idx // self.shard_size

    def _close_all_shards(self):
        for shard in self.db:
            shard.close()

    def _copy_shards(self, path):
        for shard_no in range(len(self.db)):
            shutil.copy(self._get_shard_file_path(shard_no).with_suffix(".bak"), path)
            shutil.copy(self._get_shard_file_path(shard_no).with_suffix(".dat"), path)
            shutil.copy(self._get_shard_file_path(shard_no).with_suffix(".dir"), path)

    def _make_config_file_path(self, path):
        return Path(path) / "config.json"

    def _save_config(self, path):
        with open(self._make_config_file_path(path), "wt") as f:
            json.dump(self.get_config(), f)

    def _set_config(self, num_items, shard_size, **kw_args):
        self.num_items = num_items
        self.shard_size = shard_size

    def _load_config(self, path):
        with open(self._make_config_file_path(path), "rt") as f:
            config = json.load(f)
            self._set_config(**config)
