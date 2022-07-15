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

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, Union

from tensorflow_similarity.distances import Distance
from tensorflow_similarity.types import FloatTensor


class Search(ABC):
    @abstractmethod
    def __init__(self, distance: Union[Distance, str], dim: int, verbose: bool, **kwargs):
        """Initializes a nearest neigboors search index.

        Args:
            distance: the distance used to compute the distance between
            embeddings.

            dim: the size of the embeddings.

            verbose: be verbose.
        """

    @abstractmethod
    def add(self, embedding: FloatTensor, idx: int, verbose: int = 1, **kwargs):
        """Add a single embedding to the search index.

        Args:
            embedding: The embedding to index as computed by
            the similarity model.

            idx: Embedding id as in the index table.
            Returned with the embedding to allow to lookup
            the data associated with a given embedding.

        """

    @abstractmethod
    def batch_add(self, embeddings: FloatTensor, idxs: Sequence[int], verbose: int = 1, **kwargs):
        """Add a batch of embeddings to the search index.

        Args:
            embeddings: List of embeddings to add to the index.

            idxs (int): Embedding ids as in the index table. Returned with
            the embeddings to allow to lookup the data associated
            with the returned embeddings.

            verbose: Be verbose. Defaults to 1.
        """

    @abstractmethod
    def lookup(self, embedding: FloatTensor, k: int = 5) -> Tuple[List[int], List[float]]:
        """Find embedding K nearest neighboors embeddings.

        Args:
            embedding: Query embedding as predicted by the model.
            k: Number of nearest neighboors embedding to lookup. Defaults to 5.
        """

    @abstractmethod
    def batch_lookup(self, embeddings: FloatTensor, k: int = 5) -> Tuple[List[List[int]], List[List[float]]]:
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
