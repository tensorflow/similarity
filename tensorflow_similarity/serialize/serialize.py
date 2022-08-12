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
from typing import Optional, Sequence

from tensorflow_similarity.types import FloatTensor, Tensor


class Serialize(ABC):
    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def save(
        self,
        embeddings: Sequence[FloatTensor],
        idxs: Sequence[int],
        labels: Optional[Sequence[int]] = None,
        data: Optional[Sequence[Tensor]] = None,
        verbose: int = 1,
    ) -> None:
        """Save a serialized copy of the embeddings and metadata.

        Args:
            embeddings: List of embeddings to add to the index.
            idxs: Embedding ids as in the index table. Returned with the
              embeddings to allow to lookup the data associated with the
              returned embeddings.
            labels: Embedding labels as ints.
            data: Metadata associated with the embedding, e.g., byte string
              containaing text or an image URI.
            verbose: Be verbose. Defaults to 1.
        """

    @abstractmethod
    def load(self) -> int:
        """Load index on disk

        Returns:
           Number of records reloaded.
        """
