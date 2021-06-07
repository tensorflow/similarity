from abc import ABC, abstractmethod
from typing import List, Dict, Union
from tensorflow_similarity.metrics import EvalMetric
from tensorflow_similarity.types import Lookup


class Evaluator(ABC):
    """Evaluates index performance and calibrates it.

    Index evaluators are derived from this abstract class to allow users to
    override the evaluation module to use additional data or interface
    with existing evaluation system. For example allowing to fetch data from
    a remote database.
    """

    @abstractmethod
    def evaluate(self,
                 index_size: int,
                 metrics: List[Union[str, EvalMetric]],
                 targets_labels: List[int],
                 lookups: List[List[Lookup]],
                 distance_rounding: int = 8
                 ) -> Dict[str, Union[float, int]]:
        """Evaluates lookup performances against a supplied set of metrics

        Args:
            index_size: Size of the search index.

            metrics: List of `EvalMetric()` to evaluate lookup matches against.

            targets_labels: List of the expected labels to match.

            lookups: List of lookup results as produced by the
            `Index().batch_lookup()` method.

            distance_rounding: How many digit to consider to decide if
            the distance changed. Defaults to 8.

        Returns:
            Dictionnary of metric results where keys are the metric
            names and values are the metrics values.
        """

    @abstractmethod
    def calibrate(self,
                  index_size: int,
                  calibration_metric: EvalMetric,
                  thresholds_targets: Dict[str, float],
                  targets_labels: List[int],
                  lookups: List[List[Lookup]],
                  extra_metrics: List[Union[str, EvalMetric]] = [],
                  distance_rounding: int = 8,
                  metric_rounding: int = 6,
                  verbose: int = 1):
        """Computes the distances thresholds that the calibration much match to
        meet fixed target.

        Args:
            index_size: Index size.

            calibration_metric: Metric used for calibration.

            thresholds_targets: Calibration metrics thresholds that are
            targeted. The function will find the closed distance value.

            targets_labels: List of expected labels for the lookups.

            lookup: List of lookup results as produced by the
            `Index.batch_lookup()` method.

            extra_metrics: Additional metrics that should be computed and
            reported as part of the calibration. Defaults to [].

            distance_rounding: How many digit to consider to
            decide if the distance changed. Defaults to 8.

            metric_rounding: How many digit to consider to decide if
            the metric changed. Defaults to 6.

            verbose: Be verbose. Defaults to 1.
        """
