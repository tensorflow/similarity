from abc import ABC, abstractmethod


class Matcher(ABC):

    @abstractmethod
    def add(self, embedding, idx, build=True, verbose=1):
        """Add an embedding to the index

        Args:
            embedding (tensor): Record embedding as computed
            by the model.

            idx (int): Embedding id in the index table. Used to lookup
            associated metadata.

            build (bool, optional): Rebuild the index after the addition.
            Required to make it searchable. Set to false to save time,
            Defaults to True.

            verbose (int, optional): [description]. Defaults to 1.
        """
        pass

    @abstractmethod
    def batch_add(self,  embeddings, idxs, build=True, verbose=1):
        """Add a set of record to the table

        Args:
            embedding (tensor): Record embedding as computed
            by the model.

            idxs (int): Embedding id in the index table. Used to lookup
            associated metadata.

            build (bool, optional): Rebuild the index after the addition.
            Required to make it searchable. Set to false to save time,
            Defaults to True.

            verbose (int, optional): [description]. Defaults to 1.
        """
        pass

    @abstractmethod
    def lookup(self, embedding, k=5):
        pass

    @abstractmethod
    def batch_lookup(self, embeddings, k=5):
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
