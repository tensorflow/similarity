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

from abc import abstractmethod
from abc import ABC
import math

from tensorflow_similarity.types import FloatTensor, IntTensor, BoolTensor


class RetrievalMetric(ABC):
    """Abstract base class for computing retrieval metrics.

    Attributes:
        name: Name associated with the metric object, e.g., recall@5

        canonical_name: The canonical name associated with metric,
        e.g., recall@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

        average: {'micro'} Determines the type of averaging performed over the
        queries.

            'micro': Calculates metrics globally over all queries.

            'macro': Calculates metrics for each label and takes the unweighted
                     mean.
    """
    def __init__(self,
                 name: str = '',
                 canonical_name: str = '',
                 k: int = 1,
                 distance_threshold: float = math.inf,
                 average: str = 'micro') -> None:
        self.name = name
        self.canonical_name = canonical_name
        self.k = k
        self.distance_threshold = distance_threshold
        self.average = average

        if self.k and self.k > 1:
            self.name = f'{self.name}@{k}'

        if self.distance_threshold and self.distance_threshold != 0.5:
            self.name = f'{self.name}:{self.distance_threshold}'

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "%s:%s" % (self.canonical_name, self.name)

    def get_config(self):
        return {
            "name": str(self.name),
            "canonical_name": str(self.canonical_name),
            "k": int(self.k),
            "distance_threshold": float(self.distance_threshold)
        }

    @abstractmethod
    def compute(self,
                *,
                query_labels: IntTensor,
                lookup_labels: IntTensor,
                lookup_distances: FloatTensor,
                match_mask: BoolTensor) -> FloatTensor:
        """Compute the metric

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighbor and a 0 indicates a mismatch.

        Returns:
            metric results.
        """
