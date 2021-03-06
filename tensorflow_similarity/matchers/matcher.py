from abc import ABC, abstractmethod
from tensorflow_similarity.types import FloatTensor
from typing import List, Tuple


class Matcher(ABC):

    @abstractmethod
    def add(self,
            embedding: FloatTensor,
            idx: int,
            build: bool = True,
            verbose: int = 1):
        """Add an embedding to the index

        Args:
            embedding (FloatTensor): Record embedding as computed
            by the model.

            idx (int): Embedding id in the index table. Used to lookup
            associated metadata.

            build (bool, optional): Rebuild the index after the addition.
            Required to make it searchable. Set to false to save time,
            Defaults to True.

            verbose (int, optional): [description]. Defaults to 1.
        """

    @abstractmethod
    def batch_add(self,
                  embeddings: FloatTensor,
                  idxs: List[int],
                  build: bool = True,
                  verbose: int = 1):
        """Add an embedding to the index

        Args:
            embeddings (FloatTensor): List of embeddings to add to the
            index.

            idxs (int): Embedding id in the index table. Used to lookup
            associated metadata.

            build (bool, optional): Rebuild the index after the addition.
            Required to make it searchable. Set to false to save time,
            Defaults to True.

            verbose (int, optional): [description]. Defaults to 1.
        """

    @abstractmethod
    def lookup(self,
               embedding: FloatTensor,
               k: int = 5) -> Tuple[List[int], List[float]]:
        """Find the embedding K nearest neighboors

        Args:
            embedding (FloatTensor): Target embedding as predicted by
            the model.
            k (int, optional): Number of nearest neighboors to lookup.
            Defaults to 5.
        """

    @abstractmethod
    def batch_lookup(self,
                     embeddings: FloatTensor,
                     k: int = 5) -> Tuple[List[List[int]], List[List[float]]]:
        """Find embeddings K nearest neighboors
        Args:
            embedding (FloatTensor): Target embedding as predicted by
            the model.
            k (int, optional): Number of nearest neighboors to lookup.
            Defaults to 5.
        """

    @abstractmethod
    def save(self, path: str):
        """Serializes index on disk

        Args:
            path (str): where to store the data
        """

    @abstractmethod
    def load(self, path: str):
        """load index on disk

        Args:
            path (str): where to store the data
        """
