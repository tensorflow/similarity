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

from collections.abc import MutableMapping, Sequence

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tensorflow_similarity.classification_metrics import ClassificationMetric
from tensorflow_similarity.matchers import (
    ClassificationMatch,
    make_classification_matcher,
)
from tensorflow_similarity.retrieval_metrics import RetrievalMetric
from tensorflow_similarity.retrieval_metrics.utils import compute_match_mask
from tensorflow_similarity.types import (
    CalibrationResults,
    FloatTensor,
    IntTensor,
    Lookup,
)
from tensorflow_similarity.utils import unpack_lookup_distances, unpack_lookup_labels

from .evaluator import Evaluator


class MemoryEvaluator(Evaluator):
    """In memory index performance evaluation and classification."""

    def evaluate_retrieval(
        self,
        target_labels: Sequence[int],
        lookups: Sequence[Sequence[Lookup]],
        retrieval_metrics: Sequence[RetrievalMetric],
        distance_rounding: int = 8,
    ) -> dict[str, np.ndarray]:
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
        # we also convert to np.ndarray first to avoid a slow down if
        # convert_to_tensor is called on a List.
        query_labels = tf.convert_to_tensor(np.array(target_labels))

        # data preparation: flatten and rounding
        # lookups will be shape(num_queries, num_neighbors)
        # distances will be len(num_queries x num_neighbors)
        nn_labels = unpack_lookup_labels(lookups, dtype=query_labels.dtype)
        # TODO(ovallis): The float type should be derived from the model.
        distances = unpack_lookup_distances(lookups, dtype="float32", distance_rounding=distance_rounding)

        lookup_set_size = tf.shape(nn_labels)[1]
        for m in retrieval_metrics:
            if lookup_set_size < m.k:
                raise ValueError(
                    f"Each query example returned {lookup_set_size} "
                    f"neighbors, but retrieval metric {m.name} "
                    f"requires the K >= {m.k}."
                )

        match_mask = compute_match_mask(query_labels, nn_labels)

        # compute metrics
        evaluation = {}
        for m in retrieval_metrics:
            res = m.compute(
                query_labels=query_labels,
                lookup_labels=nn_labels,
                lookup_distances=distances,
                match_mask=match_mask,
            )
            evaluation[m.name] = res.numpy()

        return evaluation

    def evaluate_classification(
        self,
        query_labels: IntTensor,
        lookup_labels: IntTensor,
        lookup_distances: FloatTensor,
        distance_thresholds: FloatTensor,
        metrics: Sequence[ClassificationMetric],
        matcher: str | ClassificationMatch,
        distance_rounding: int = 8,
        verbose: int = 1,
    ) -> dict[str, np.ndarray]:
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
        matcher = make_classification_matcher(matcher)
        matcher.compile(distance_thresholds=distance_thresholds)

        # compute the tp, fp, tn, fn counts
        matcher.compute_count(
            query_labels=query_labels,
            lookup_labels=lookup_labels,
            lookup_distances=lookup_distances,
        )

        # evaluating performance as distance value increase
        if verbose:
            pb = tqdm(total=len(metrics), desc="Evaluating")

        # evaluating performance as distance value increase
        results: dict[str, np.ndarray] = {"distance": distance_thresholds.numpy()}
        for m in metrics:
            res = m.compute(
                tp=matcher.tp,
                fp=matcher.fp,
                tn=matcher.tn,
                fn=matcher.fn,
                count=matcher.count,
            )
            results[m.name] = res.numpy()

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        return results

    def calibrate(
        self,
        target_labels: Sequence[int],
        lookups: Sequence[Sequence[Lookup]],
        thresholds_targets: MutableMapping[str, float],
        calibration_metric: ClassificationMetric,
        matcher: str | ClassificationMatch,
        extra_metrics: Sequence[ClassificationMetric] = [],
        distance_rounding: int = 8,
        metric_rounding: int = 6,
        verbose: int = 1,
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

            extra_metrics: Additional classification metrics that should be
            computed and reported as part of the calibration. Defaults to [].

            distance_rounding: How many digit to consider to
            decide if the distance changed. Defaults to 8.

            metric_rounding: How many digit to consider to decide if
            the metric changed. Defaults to 6.

            verbose: Be verbose. Defaults to 1.
        Returns:
            CalibrationResults containing the thresholds and cutpoints Dicts.

        Raises:
            ValueError: lookupds must not be empty.
        """
        # TODO (ovallis): Assert if index is empty, or if the lookup is empty.
        if len(lookups) == 0:
            raise ValueError("lookups must not be empty. Is there no data in the index?")

        # making a single list of metrics
        # Need expl covariance problem
        combined_metrics = list(extra_metrics)
        combined_metrics.append(calibration_metric)

        # we also convert to np.ndarray first to avoid a slow down if
        # convert_to_tensor is called on a List.
        query_labels = tf.convert_to_tensor(np.array(target_labels))

        # data preparation: flatten and rounding
        # lookups will be shape(num_queries, num_neighbors)
        # distances will be len(num_queries x num_neighbors)
        # TODO(ovallis): The float type should be derived from the model.
        lookup_distances = unpack_lookup_distances(lookups, dtype="float32", distance_rounding=distance_rounding)
        lookup_labels = unpack_lookup_labels(lookups, dtype=query_labels.dtype)

        # the unique set of distance values sorted ascending
        unique_distances, _ = tf.unique(tf.reshape(lookup_distances, (-1)))
        distance_thresholds = tf.sort(unique_distances)

        results = self.evaluate_classification(
            query_labels=query_labels,
            lookup_labels=lookup_labels,
            lookup_distances=lookup_distances,
            distance_thresholds=distance_thresholds,
            metrics=combined_metrics,
            matcher=matcher,
            distance_rounding=distance_rounding,
            verbose=verbose,
        )

        cutpoints: dict[str, dict[str, str | float]] = {}

        cutpoints["optimal"] = self._optimal_cutpoint(results, calibration_metric)

        for name, value in thresholds_targets.items():
            target_cp = self._target_cutpoints(results, calibration_metric, name, value)
            if target_cp:
                cutpoints[name] = target_cp

        # Add the calibration metric as 'value' in the thresholds dict.
        results["value"] = results[calibration_metric.name]

        return CalibrationResults(cutpoints=cutpoints, thresholds=results)

    def _optimal_cutpoint(
        self,
        metrics: dict[str, np.ndarray],
        calibration_metric: ClassificationMetric,
    ) -> dict[str, str | float]:
        """Compute the optimal distance threshold for the calibration metric.

        Args:
            metrics: A mapping from metric name to a list of metric values
              computed for each unique distance observed in the calibration
              dataset. This dict should also include a distance key mapping
              to the list of observed distances. We expect the distances to be
              ascending, and for all lists to be the same length where each
              index in the list corresponds to a distance in the distance list.
            calibration_metric: The ClassificationMetric used for calibration.

        Returns:
            A Dict of the metric values at the calibrated distance. This also
              includes the cutpoint name, the distance, and the value of the
              calibration metric.

              ```
              {
                  'name': 'optimal', # Cutpoint name
                  'value': 0.99,     # Calibration metric at the cutpoint
                  'distance': 0.1,   # Calibrated distance
                  'precision': 0.99, # Here, we calibrated using precision
                  'f1': 0.4,         # We also computed F1 at this point
              }
              ```
        """
        if calibration_metric.maximize:
            idx = self._last_argmax(metrics[calibration_metric.name])
        else:
            idx = self._last_argmin(metrics[calibration_metric.name])

        optimal_cp = {
            "name": "optimal",
            "value": metrics[calibration_metric.name][idx].item(),
        }
        for metric_name in metrics.keys():

            optimal_cp[metric_name] = metrics[metric_name][idx].item()

        return optimal_cp

    def _target_cutpoints(
        self,
        metrics: dict[str, np.ndarray],
        calibration_metric: ClassificationMetric,
        target_name: str,
        target_value: float,
    ) -> dict[str, str | float]:
        """Compute the distance at the target metric for the calibration metric.

        Args:
            metrics: A mapping from metric name to a list of metric values
              computed for each unique distance observed in the calibration
              dataset. This dict should also include a distance key mapping
              to the list of observed distances. We expect the distances to be
              ascending, and for all lists to be the same length where each
              index in the list corresponds to a distance in the distance list.
            calibration_metric: The ClassificationMetric used for calibration.
            target_name: The name for the target cutpoint.
            target_value: The target metric value.

        Returns:
            A Dict of the metric values at the calibrated distance. This also
              includes the cutpoint name, the distance, and the value of the
              calibration metric.

              ```
              {
                  'name': '0.90',     # Target cutpoint name.
                  'value': 0.901,     # Closest metric value at or above the
                                      # target cutpoint, assuming we are
                                      # maximizing the calibration metric.
                  'distance': 0.1,    # Calibrated distance.
                  'precision': 0.901, # Here, we calibrated using precision.
                  'f1': 0.4,          # We also computed F1 at this point.
              }
              ```
        """
        indicators = np.where(metrics[calibration_metric.name] >= target_value)[0]
        target_cp: dict[str, str | float] = {}

        if indicators.size > 0:
            if calibration_metric.increasing:
                # Take the first index above the target if the metric is increasing
                idx = indicators[0]
            else:
                # Take the last index above the target if the metric is decreasing
                idx = indicators[-1]

            target_cp["name"] = target_name
            target_cp["value"] = metrics[calibration_metric.name][idx].item()
            for metric_name in metrics.keys():
                target_cp[metric_name] = metrics[metric_name][idx].item()

        return target_cp

    def _last_argmax(self, x: np.ndarray) -> int:
        """The index of the last occurrence of the max value.

        In case of multiple occurrences of the maximum values, the index
        corresponding to the last occurrence is returned.

        Args:
            A 1D np.ndarray or List[float].

        Returns:
            The index of the last occurrence of the max value.
        """
        revx = x[::-1]
        return (len(x) - np.argmax(revx) - 1).item()

    def _last_argmin(self, x: np.ndarray) -> int:
        """The index of the last occurrence of the min value.

        In case of multiple occurrences of the minimum values, the index
        corresponding to the last occurrence is returned.

        Args:
            A 1D np.ndarray or List[float].

        Returns:
            The index of the last occurrence of the min value.
        """
        revx = x[::-1]
        return (len(x) - np.argmin(revx) - 1).item()

    def _is_lower(self, curr, prev, equal=False):
        if equal:
            return curr <= prev
        return curr < prev

    def _is_higher(self, curr, prev, equal=False):
        if equal:
            return curr >= prev
        return curr > prev
