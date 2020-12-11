from abc import ABC, abstractmethod


class Table(ABC):

    @abstractmethod
    def add(self, embedding, label=None, data=None):
        """Add a record to the table

        Args:
            embedding (tensor): Record embedding as computed
            by the model. Defaults to None.

            label (int, optional): Class numerical id. Defaults to None.

            data (tensor, optional): Record data. Defaults to None.

        Returns:
            int: associated record id
        """
        pass

    @abstractmethod
    def batch_add(self,  embeddings, labels=None, data=None):
        """Add a set of record to the table

        Args:
            embeddings (list(tensor)): Record embedding as computed
            by the model.

            labels (list(int), optional): Class numerical id. Defaults to None.

            datas (list(tensor), optional): Record data. Defaults to No.

        See:
            add() for what a record contains.

        Returns:
            list(int): list of associated record id
        """
        pass

    @abstractmethod
    def get(self, idx):
        """Get record from the table

        Args:
            idx (int): record_id to lookup

        Returns:
            list: data associated with the record_id
        """
        pass

    @abstractmethod
    def batch_get(self, idxs):
        """Get records from the table

        Args:
            idxs (int): record ids to lookup

        Returns:
            list(lists): data associated with the record ids
        """
        pass

    @abstractmethod
    def size(self):
        "Number of record in the table"
        pass

    @abstractmethod
    def save(self, path):
        """Serializes index on disk

        Args:
            path (str): where to store the data
        """
        pass

    @abstractmethod
    def load(self, path):
        """load index on disk

        Args:
            path (str): where to store the data
        """
        pass
