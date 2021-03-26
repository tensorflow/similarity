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
        """Add a single embedding to the matcher.

        Args:
            embedding: The embedding to index as computed by
            the similarity model.

            idx: Embedding id as in the index table.
            Returned with the embedding to allow to lookup
            the data associated with a given embedding.

            build: Rebuild the index after the addition.
            Required to make the embedding searchable.
            Set to false to save time between successive addition.
            Defaults to True.
        """

    @abstractmethod
    def batch_add(self,
                  embeddings: FloatTensor,
                  idxs: List[int],
                  build: bool = True,
                  verbose: int = 1):
        """Add a batch of embeddings to the matcher.

        Args:
            embeddings: List of embeddings to add to the index.

            idxs (int): Embedding ids as in the index table. Returned with
            the embeddings to allow to lookup the data associated
            with the returned embeddings.

            build: Rebuild the index after the addition. Required to
            make the embeddings searchable. Set to false to save
            time between successive addition. Defaults to True.

            verbose: Be verbose. Defaults to 1.
        """

    @abstractmethod
    def lookup(self,
               embedding: FloatTensor,
               k: int = 5) -> Tuple[List[int], List[float]]:
        """Find embedding K nearest neighboors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """

    @abstractmethod
    def batch_lookup(self,
                     embeddings: FloatTensor,
                     k: int = 5) -> Tuple[List[List[int]], List[List[float]]]:
        """Find embeddings K nearest neighboors embeddings.

        Args:
            embedding: Batch of query embeddings as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """

    @abstractmethod
    def save(self, path: str):
        """Serializes the index data on disk

        Args:
            path: where to store the data
        """

    @abstractmethod
    def load(self, path: str):
        """load index on disk

        Args:
            path: where to store the data
        """
