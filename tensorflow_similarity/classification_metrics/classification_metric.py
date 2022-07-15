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

from tensorflow_similarity.types import FloatTensor


class ClassificationMetric(ABC):
    """Abstract base class for computing classification metrics.

    Args:
        name: Name associated with a specific metric object, e.g.,
        accuracy@0.1

        canonical_name: The canonical name associated with metric, e.g.,
        accuracy

        maximize: If True, we attempt to maximize the calibration metric
        as the distance increases.

        increasing: If True, the metric should increase as the distance
        increases.

    `ClassificationMetric` measure the matching classification between the
    query label and the label derived from the set of lookup results.

    The `compute()` method supports computing the metric for a set of values,
    where each value represents the counts at a specific distance threshold.
    """

    def __init__(
        self, name: str = "", canonical_name: str = "", maximize: bool = True, increasing: bool = True
    ) -> None:
        self.name = name
        self.canonical_name = canonical_name
        self.maximize = maximize
        self.increasing = increasing

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "%s:%s" % (self.canonical_name, self.name)

    def get_config(self):
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "maximize": self.maximize,
            "increasing": self.increasing,
        }

    @abstractmethod
    def compute(self, tp: FloatTensor, fp: FloatTensor, tn: FloatTensor, fn: FloatTensor, count: int) -> FloatTensor:
        """Compute the classification metric.

        Args:
            tp: A 1D FloatTensor containing the count of True Positives at each
            distance threshold.

            fp: A 1D FloatTensor containing the count of False Positives at
            each distance threshold.

            tn: A 1D FloatTensor containing the count of True Negatives at each
            distance threshold.

            fn: A 1D FloatTensor containing the count of False Negatives at
            each distance threshold.

            count: The total number of queries

        Returns:
            A 1D FloatTensor containing the metric at each distance threshold.
        """
