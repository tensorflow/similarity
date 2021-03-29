import numpy as np
from pathlib import Path
from tensorflow_similarity.types import FloatTensor, List, Tuple, Tensor
from tensorflow_similarity.types import Optional, PandasDataFrame
import pandas as pd
from .table import Table


class MemoryTable(Table):
    """Efficient in-memory dataset table"""

    def __init__(self) -> None:
        # We are using a native python array in memory for its row speed.
        # Serialization / export relies on Arrow.
        self.labels: List[Optional[int]] = []
        self.embeddings: List[FloatTensor] = []
        self.data: List[Optional[Tensor]] = []
        self.num_items: int = 0
        pass

    def add(self,
            embedding: FloatTensor,
            label: Optional[int] = None,
            data: Optional[Tensor] = None) -> int:
        """Add an Embedding record to the table.

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
            embeddings: List[FloatTensor],
            labels: List[Optional[int]] = None,
            data: List[Optional[Tensor]] = None) -> List[int]:
        """Add a set of embedding records to the table.

        Args:
            embeddings: Embeddings predicted by the model.

            labels: Class numerical ids. Defaults to None.

            data: Data associated with the embeddings. Defaults to None.

        See:
            add() for what a record contains.

        Returns:
            List of associated record id.
        """

        idxs: List[int] = []
        for idx, embedding in enumerate(embeddings):
            label = None if labels is None else labels[idx]
            rec_data = None if data is None else data[idx]
            idxs.append(self.add(embedding, label, rec_data))
        return idxs

    def get(self, idx: int) -> Tuple[FloatTensor,
                                     Optional[int],
                                     Optional[Tensor]]:
        """Get an embedding record from the table

        Args:
            idx: Id of the record to fetch.

        Returns:
            record associated with the requested id.
        """

        return self.embeddings[idx], self.labels[idx], self.data[idx]

    def batch_get(self, idxs: List[int]
                  ) -> Tuple[List[FloatTensor], List[Optional[int]],
                             List[Optional[Tensor]]]:
        """Get embedding records from the table

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
        "Number of record in the table."
        return self.num_items

    def save(self, path: str, compression: bool = True) -> None:
        """Serializes index on disk.

        Args:
            path: where to store the data.
            compression: Compress index data. Defaults to True.
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

    def load(self, path: str) -> int:
        """load index on disk

        Args:
            path: which directory to use to store the index data.

        Returns:
           Number of records reloaded.
        """

        fname = self._make_fname(path, check_file_exit=True)
        data = np.load(fname, allow_pickle=True)
        self.embeddings = list(data['embeddings'])
        self.labels = list(data['labels'])
        self.data = list(data['data'])
        self.num_items = len(self.embeddings)
        print("loaded %d records from %s" % (self.size(), path))
        return self.size()

    def _make_fname(self, path: str, check_file_exit: bool = False) -> str:
        p = Path(path)
        if not p.exists():
            raise ValueError("Index path doesn't exist")
        fname = p / 'index.npz'

        # only for loading
        if check_file_exit and not fname.exists():
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
            "lables": self.labels[:num_records]
        }

        # forcing type from Any to PandasFrame
        df: PandasDataFrame = pd.DataFrame.from_dict(data)
        return df
