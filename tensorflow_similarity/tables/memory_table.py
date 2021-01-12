import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tensorflow_similarity.types import FloatTensorLike

from .table import Table


class MemoryTable(Table):
    """Efficient in-memory dataset table powered by Apache Arrrow"""

    def __init__(self) -> None:
        # We are using a native python array in memory for its row speed.
        # Serialization / export relies on Arrow.
        self.labels: List[Optional[int]] = []
        self.embeddings: List[FloatTensorLike] = []
        self.data: List[Optional[FloatTensorLike]] = []
        self.num_items: int = 0
        pass

    def add(self,
            embedding: FloatTensorLike,
            label: Optional[int] = None,
            data: Optional[FloatTensorLike] = None) -> int:
        """Add a record to the table

        Args:
            embedding (tensor): Record embedding as computed
            by the model.

            label (int, optional): Class numerical id. Defaults to None.

            data (tensor, optional): Record data. Defaults to None.

        Returns:
            int: associated record id
        """
        idx = self.num_items
        self.labels.append(label)
        self.embeddings.append(embedding)
        self.data.append(data)
        self.num_items += 1
        return idx

    def batch_add(
            self,
            embeddings: List[FloatTensorLike],
            labels: List[Optional[int]] = None,
            data: List[Optional[FloatTensorLike]] = None) -> List[int]:
        """Add a set of record to the mapper

        Args:
            embeddings (list(tensor)): Record embedding as computed
            by the model.

            labels (list(int), optional): Class numerical id. Defaults to None.

            datas (list(tensor), optional): Record data. Defaults to No.

        Returns:
            list(int): list of associated record id
        """
        idxs: List[int] = []
        for idx, embedding in enumerate(embeddings):
            label = None if labels is None else labels[idx]
            rec_data = None if data is None else data[idx]
            idxs.append(self.add(embedding, label, rec_data))
        return idxs

    def get(self, idx: int) -> Tuple[FloatTensorLike,
                                     Optional[int],
                                     Optional[FloatTensorLike]]:
        """Get record from the mapper

        Args:
            idx (int): record_id to lookup

        Returns:
            record: record associated with the record_id
        """
        return self.embeddings[idx], self.labels[idx], self.data[idx]

    def batch_get(self, idxs: List[int]
                  ) -> Tuple[List[FloatTensorLike], List[Optional[int]],
                             List[Optional[FloatTensorLike]]]:
        """Get records from the table

        Args:
            idx (int): record_id to lookup

        Returns:
            list(records): record associated with the record_id
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
        "Number of record in the mapper"
        return self.num_items

    def save(self, path: str, compression: bool = True) -> None:
        """Serializes index on disk

        Args:
            path (str): where to store the data
        """
        fname = self._make_fname(path)
        if compression:
            np.savez_compressed(fname,
                                embeddings=self.embeddings,
                                labels=self.labels,
                                data=self.data)
        else:
            np.savez(fname,
                     embeddings=self.embeddings,
                     labels=self.labels,
                     data=self.data)

    def load(self, path: str) -> None:
        """load index on disk

        Args:
            path (str): where to store the data
        """

        fname = self._make_fname(path, check_file_exit=True)
        data = np.load(fname, allow_pickle=True)
        self.embeddings = list(data['embeddings'])
        self.labels = list(data['labels'])
        self.data = list(data['data'])
        self.num_items = len(self.embeddings)
        print("loaded %d records from %s" % (self.size(), path))

    def _make_fname(self, path: str, check_file_exit: bool = False) -> str:
        p = Path(path)
        if not p.exists():
            raise ValueError("Index path doesn't exist")
        fname = p / 'index.npz'

        # only for loading
        if check_file_exit and not fname.exists():
            raise ValueError("Index file not found")
        return str(fname)
