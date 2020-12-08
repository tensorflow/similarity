import numpy as np
from pathlib import Path

from .table import Table


class MemoryTable(Table):
    """Efficient in-memory dataset table powered by Apache Arrrow"""

    def __init__(self):
        # We are using a native python array in memory for its row speed.
        # Serialization / export relies on Arrow.
        self.labels = []
        self.embeddings = []
        self.data = []
        self.num_items = 0
        pass

    def add(self, embedding, label=None, data=None):
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

    def batch_add(self, embeddings, labels=None, data=None):
        """Add a set of record to the mapper

        Args:
            embeddings (list(tensor)): Record embedding as computed
            by the model.

            labels (list(int), optional): Class numerical id. Defaults to None.

            datas (list(tensor), optional): Record data. Defaults to No.

        Returns:
            list(int): list of associated record id
        """
        idxs = []
        for idx, embedding in enumerate(embeddings):
            label = labels[idx] if labels else None
            rec_data = data[idx] if data else None
            idxs.append(self.add(embedding, label, rec_data))
        return idxs

    def get(self, idx):
        """Get record from the mapper

        Args:
            idx (int): record_id to lookup

        Returns:
            record: record associated with the record_id
        """
        return self.embeddings[idx], self.labels[idx], self.data[idx]

    def batch_get(self, idxs):
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

    def dump(self):
        """Returns all the elements in the index

        This is needed to rebuild the indexer on reload.
        """
        return self.embeddings, self.labels, self.data

    def size(self):
        "Number of record in the mapper"
        return self.num_items

    def save(self, path, compression=True):
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

    def load(self, path):
        """load index on disk

        Args:
            path (str): where to store the data
        """

        fname = self._make_fname(path, check_file_exit=True)
        data = np.load(fname)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.data = data['data']

        # ! Code assume the counter is one ahead
        self.num_items = len(self.embeddings)
        print("loaded %d records from %s" % (self.size(), path))

    def _make_fname(self, path, check_file_exit=False):
        p = Path(path)
        if not p.exists():
            raise ValueError("Index path doesn't exist")
        fname = p / 'index.npz'

        # only for loading
        if check_file_exit and not fname.exists():
            raise ValueError("Index file not found")
        return str(fname)
