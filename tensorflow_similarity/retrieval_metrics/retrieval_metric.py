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
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import tensorflow as tf

from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor


class RetrievalMetric(ABC):
    """Abstract base class for computing retrieval metrics.

    Args:
        name: Name associated with the metric object, e.g., recall@5

        canonical_name: The canonical name associated with metric, e.g.,
        recall@K

        k: The number of nearest neighbors over which the metric is computed.

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

        average: {'micro'} Determines the type of averaging performed over the
        queries.

        * 'micro': Calculates metrics globally over all queries.
        * 'macro': Calculates metrics for each label and takes the unweighted
        mean.

        drop_closest_lookup: If True, remove the closest lookup before computing
        the metrics. This is used when the query set == indexed set.

    `RetrievalMetric` measure the retrieval quality given a query label and the
    labels from the set of lookup results.
    """

    def __init__(
        self,
        name: str = "",
        canonical_name: str = "",
        k: int = 5,
        distance_threshold: float = math.inf,
        average: str = "micro",
        drop_closest_lookup: bool = False,
    ) -> None:
        self._name = name
        self.canonical_name = canonical_name
        self.k = k
        self.distance_threshold = distance_threshold
        self.average = average
        self.drop_closest_lookup = drop_closest_lookup

    @property
    def name(self) -> str:
        if self.distance_threshold and self.distance_threshold != math.inf:
            return f"{self._name}@{self.k} : distance_threshold@{self.distance_threshold}"
        else:
            return f"{self._name}@{self.k}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "%s : %s" % (self.canonical_name, self.name)

    def get_config(self):
        return {
            "name": str(self.name),
            "canonical_name": str(self.canonical_name),
            "k": int(self.k),
            "distance_threshold": float(self.distance_threshold),
        }

    @abstractmethod
    def compute(
        self,
        *,  # keyword only arguments see PEP-570
        query_labels: IntTensor,
        lookup_labels: IntTensor,
        lookup_distances: FloatTensor,
        match_mask: BoolTensor,
    ) -> FloatTensor:
        """Compute the retrieval metric.

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

            match_mask: A 2D mask where a 1 indicates a match between the
            jth query and the kth neighbor and a 0 indicates a mismatch.

        Returns:
            A rank 0 tensor containing the metric.
        """

    def _check_shape(self, query_labels: IntTensor, match_mask: FloatTensor) -> None:
        if tf.shape(match_mask)[1] < self.k:
            raise ValueError(
                f"The number of neighbors must be >= K. Number of neighbors is"
                f" {tf.shape(match_mask)[1]} but K is {self.k}."
            )

        if tf.shape(match_mask)[0] != tf.shape(query_labels)[0]:
            raise ValueError(
                "The number of lookup sets must equal the number of query "
                f"labels. Number of lookup sets is {tf.shape(match_mask)[0]} "
                "but the number of query labels is "
                f"{tf.shape(query_labels)[0]}."
            )
