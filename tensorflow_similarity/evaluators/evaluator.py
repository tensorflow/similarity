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
from typing import Dict, MutableMapping, Sequence, Union

import numpy as np

from tensorflow_similarity.classification_metrics import ClassificationMetric
from tensorflow_similarity.matchers import ClassificationMatch
from tensorflow_similarity.retrieval_metrics import RetrievalMetric
from tensorflow_similarity.types import (
        Lookup, CalibrationResults, IntTensor, FloatTensor)


class Evaluator(ABC):
    """Evaluates search index performance and calibrates it.

    Index evaluators are derived from this abstract class to allow users to
    override the evaluation module to use additional data or interface
    with existing evaluation system. For example allowing to fetch data from
    a remote database.
    """

    @abstractmethod
    def evaluate_retrieval(
            self,
            target_labels: Sequence[int],
            lookups: Sequence[Sequence[Lookup]],
            retrieval_metrics: Sequence[RetrievalMetric],
            distance_rounding: int = 8) -> Dict[str, np.ndarray]:
        """Evaluates lookup performances against a supplied set of metrics

        Args:
            target_labels: Sequence of the expected labels to match.

            lookups: Sequence of lookup results as produced by the
            `Index().batch_lookup()` method.

            retrieval_metrics: Sequence of `RetrievalMetric()` to evaluate
            lookup matches against.

            distance_rounding: How many digit to consider to decide if
            the distance changed. Defaults to 8.

        Returns:
            Dictionary of metric results where keys are the metric names and
            values are the metrics values.
        """

    @abstractmethod
    def evaluate_classification(
        self,
        query_labels: IntTensor,
        lookup_labels: IntTensor,
        lookup_distances: FloatTensor,
        distance_thresholds: FloatTensor,
        metrics: Sequence[ClassificationMetric],
        matcher: Union[str, ClassificationMatch],
        distance_rounding: int = 8,
        verbose: int = 1
    ) -> Dict[str, np.ndarray]:
        """Evaluate the classification performance.

        Compute the classification metrics given a set of queries, lookups, and
        distance thresholds.

        Args:
            query_labels: Sequence of expected labels for the lookups.

            lookup_labels: A 2D tensor where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D tensor where the jth row is the distances
            between the jth query and the set of k neighbors.

            distance_thresholds: A 1D tensor denoting the distances points at
            which we compute the metrics.

            metrics: The set of classification metrics.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            distance_rounding: How many digit to consider to
            decide if the distance changed. Defaults to 8.

            verbose: Be verbose. Defaults to 1.
        Returns:
            A Mapping from metric name to the list of values computed for each
            distance threshold.
        """

    @abstractmethod
    def calibrate(
        self,
        target_labels: Sequence[int],
        lookups: Sequence[Sequence[Lookup]],
        thresholds_targets: MutableMapping[str, float],
        calibration_metric: ClassificationMetric,
        matcher: Union[str, ClassificationMatch],
        extra_metrics: Sequence[ClassificationMetric] = [],
        distance_rounding: int = 8,
        metric_rounding: int = 6,
        verbose: int = 1
    ) -> CalibrationResults:
        """Computes the distances thresholds that the classification must match to
        meet a fixed target.

        Args:
            target_labels: Sequence of expected labels for the lookups.

            lookup: Sequence of lookup results as produced by the
            `Index.batch_lookup()` method.

            thresholds_targets: classification metrics thresholds that are
            targeted. The function will find the closed distance value.

            calibration_metric: Classification metric used for calibration.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            extra_metrics: Additional metrics that should be computed and
            reported as part of the classification. Defaults to [].

            distance_rounding: How many digit to consider to
            decide if the distance changed. Defaults to 8.

            metric_rounding: How many digit to consider to decide if
            the metric changed. Defaults to 6.

            verbose: Be verbose. Defaults to 1.
        Returns:
            CalibrationResults containing the thresholds and cutpoints Dicts.
        """
