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

from typing import Tuple

from .classification_match import ClassificationMatch
from tensorflow_similarity.types import FloatTensor, IntTensor


class MatchNearest(ClassificationMatch):
    """Match metrics for labels at k=1."""

    def __init__(self,
                 name: str = 'nearest',
                 **kwargs) -> None:

        if 'canonical_name' not in kwargs:
            kwargs['canonical_name'] = 'match_nearest'

        super().__init__(name=name, **kwargs)

    def derive_match(self,
                     lookup_labels: IntTensor,
                     lookup_distances: FloatTensor
                     ) -> Tuple[IntTensor, FloatTensor]:
        """Derive a match label and distance from a set of K neighbors.

        For each query, derive a single match label and distance given the
        associated set of lookup labels and distances.

        Args:
            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of FloatTensors:
                derived_labels: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the derived
                label for the jth query.

                derived_distances: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the distance
                associated with the jth derived label.
        """

        return lookup_labels[:, :1], lookup_distances[:, :1]
