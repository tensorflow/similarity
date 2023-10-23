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
"""Vectorized embedding pairwise distances computation functions"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import FloatTensor


class Distance(ABC):
    """Abstract class for distance computation."""

    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    def call(self, query_embeddings: FloatTensor, key_embeddings: FloatTensor) -> FloatTensor:
        """Compute pairwise distances for a given batch.

        Args:
            query_embeddings: Embeddings to compute the pairwise one.
            key_embeddings: Embeddings to compute the pairwise one.

        Returns:
            FloatTensor: Pairwise distance tensor.
        """

    def __call__(self, query_embeddings: FloatTensor, key_embeddings: FloatTensor):
        return self.call(query_embeddings, key_embeddings)

    def __str__(self) -> str:
        return self.name

    def get_config(self) -> dict[str, Any]:
        """Contains the distance configuration.

        Returns:
            A Python dict containing the configuration of the distance obj.
        """
        config = {"name": self.name}

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Distance:
        """Build a distance from a config.

        Args:
            config: A Python dict containing the configuration of the distance.

        Returns:
            A distance instance.
        """
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when deserializing '{cls.__name__}' using" f"config={config}.\n\nException encountered: {e}"
            )
