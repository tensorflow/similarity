from typing import Dict, Type, Union

from .classification_metric import ClassificationMetric  # noqa
from .f1_score import F1Score  # noqa
from .false_positive_rate import FalsePositiveRate  # noqa
from .negative_predictive_value import NegativePredictiveValue  # noqa
from .precision import Precision  # noqa
from .recall import Recall  # noqa
from .accuracy import Accuracy  # noqa


def make_classification_metric(
        metric: Union[str, 'ClassificationMetric']) -> 'ClassificationMetric':
    """Convert classification metric from str name to object if needed.

    Args:
        metric: ClassificationMetric() or metric name.

    Raises:
        ValueError: metric name is invalid.

    Returns:
        ClassificationMetric: Instantiated metric if needed.
    """
    # ! Metrics must be non-instantiated.
    METRICS_ALIASES: Dict[str, Type['ClassificationMetric']] = {
        "recall": Recall,
        "precision": Precision,
        "f1": F1Score,
        "f1score": F1Score,
        "f1_score": F1Score,
        "acc": Accuracy,
        "accuracy": Accuracy,
        "npv": NegativePredictiveValue,
        "negative_predicitve_value": NegativePredictiveValue,
        "fpr": FalsePositiveRate,
        "false_positive_rate": FalsePositiveRate,
    }

    if isinstance(metric, str):
        if metric.lower() in METRICS_ALIASES:
            metric = METRICS_ALIASES[metric.lower()]()
        else:
            raise ValueError(f'Unknown metric name: {metric}, typo?')

    return metric
