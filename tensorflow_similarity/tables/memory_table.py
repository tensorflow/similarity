import numpy as np
from pathlib import Path
from tensorflow_similarity.types import FloatTensorLike, List, Tuple
from tensorflow_similarity.types import Optional, PandasDataFrame
import pandas as pd
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
            embedding (FloatTensorLike): Record embedding predicted
            by the model.

            label (int, optional): Class numerical id. Defaults to None.

            data (FloatTensorLike, optional): Record data. Defaults to None.

        Returns:
            int: associated record id.
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
        """Add a set of record to the table

        Args:
            embeddings (FloatTensorLike): Record the embeddings predicted
            by the model.

            labels (list(int), optional): Class numerical id. Defaults to None.

            datas (list(FloatTensorLike), optional): Record data.
            Defaults to None.

        See:
            add() for what a record contains.

        Returns:
            list(int): list of associated record id.
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
            idx (int): lookup record id to fetch

        Returns:
            record: record associated with the requested record id
        """
        return self.embeddings[idx], self.labels[idx], self.data[idx]

    def batch_get(self, idxs: List[int]
                  ) -> Tuple[List[FloatTensorLike], List[Optional[int]],
                             List[Optional[FloatTensorLike]]]:
        """Get records from the table

        Args:
            idxs (List[int]): lookups record ids to fetch

        Returns:
            Tuple(List): data associated with the requested record ids
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
        """Serializes index on disks

        Args:
            path (str): where to store the data.
            compression (bool): Compress index data. Defaults to True.
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

    def to_data_frame(self, num_items: int = 0) -> PandasDataFrame:
        """Export data as pandas dataframe

        Args:
            num_items (int, optional): Num items to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """
        if not num_items:
            num_items = self.num_items

        data = {
            "embeddings": self.embeddings[:num_items],
            "data": self.data[:num_items],
            "lables": self.labels[:num_items]
        }

        # forcing type from Any to PandasFrame
        df: PandasDataFrame = pd.DataFrame.from_dict(data)
        return df
