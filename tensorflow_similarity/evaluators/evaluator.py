from abc import ABC, abstractmethod
from typing import List, Dict, Union
from tensorflow_similarity.metrics import EvalMetric


class Evaluator(ABC):
    """Evaluate index performance and calibrates it

    Note: Evaluators are derived from this abstract class to allow users to
    override the evaluation to use additional data or interface with different
    evaluation system. For example fetching data from a remote database.
    """

    @abstractmethod
    def evaluate(self,
                 index_size: int,
                 metrics: List[Union[str, EvalMetric]],
                 targets_labels: List[int],
                 lookups: List[List[Dict[str, Union[float, int]]]],
                 distance_rounding: int = 8
                 ) -> Dict[str, Union[float, int]]:
        """Evaluate lookup performances against a supplied set of metrics

        Args:
            index_size (int): Size of the search index.

            metrics (List[Union[str, EvalMetric]]): List of `EvalMetric()` to
            evaluate lookup matches against.

            targets_labels (List[int]): List of expected matched labels.

            lookups (List[List[Dict[str, Union[float, int]]]]): List of lookup
            results as produced by the `Index()` `batch_lookup()` method.

            distance_rounding (int, optional): How many digit to consider to
            decide if the distance changed. Defaults to 8.

        Returns:
            Dict[str, Union[float, int]]: Dictionnary of metric results where
            keys are the metric names and values are the metrics values.
        """

    @abstractmethod
    def calibrate(self,
                  index_size: int,
                  calibration_metric: EvalMetric,
                  thresholds_targets: Dict[str, float],
                  targets_labels: List[int],
                  lookups: List[List[Dict[str, Union[float, int]]]],
                  extra_metrics: List[Union[str, EvalMetric]] = [],
                  distance_rounding: int = 8,
                  metric_rounding: int = 6,
                  verbose: int = 1):
        """Compute the distances thresholds that match specified metrics
        targets.

        Args:
            index_size (int): Index size.

            calibration_metric (EvalMetric): Metric used for calibration.

            thresholds_targets (Dict[str, float]): Calibration metrics
            thresholds that are targeted. The function will find the closed
            distance value.

            targets_labels (List[int]): List of expected labels for
            the lookups.

            lookups (List[List[Dict[str, Union[float, int]]]]): List of lookup
            results as produced by the `Index()` `batch_lookup()` method.

            extra_metrics (List[Union[str, EvalMetric]], optional): Additional
            metrics that should be computed and reported as part of the
            calibration. Defaults to [].

            distance_rounding (int, optional): How many digit to consider to
            decide if the distance changed. Defaults to 8.

            metric_rounding (int, optional): [description]. How many digit to
            consider to decide if the metric changed. Defaults to 6.

            verbose (int, optional): Be verbose. Defaults to 1.
        """
