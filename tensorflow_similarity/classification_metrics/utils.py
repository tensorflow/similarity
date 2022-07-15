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

from typing import Dict, Type, Union

from .binary_accuracy import BinaryAccuracy  # noqa
from .classification_metric import ClassificationMetric  # noqa
from .f1_score import F1Score  # noqa
from .false_positive_rate import FalsePositiveRate  # noqa
from .negative_predictive_value import NegativePredictiveValue  # noqa
from .precision import Precision  # noqa
from .recall import Recall  # noqa


def make_classification_metric(metric: Union[str, ClassificationMetric], name: str = "") -> ClassificationMetric:
    """Convert classification metric from str name to object if needed.

    Args:
        metric: ClassificationMetric() or metric name.

    Raises:
        ValueError: Unknown metric name: {metric}, typo?

    Returns:
        ClassificationMetric: Instantiated metric if needed.
    """
    # ! Metrics must be non-instantiated.
    METRICS_ALIASES: Dict[str, Type[ClassificationMetric]] = {
        "recall": Recall,
        "precision": Precision,
        "f1": F1Score,
        "f1score": F1Score,
        "f1_score": F1Score,
        "binary_accuracy": BinaryAccuracy,
        "npv": NegativePredictiveValue,
        "negative_predicitve_value": NegativePredictiveValue,
        "fpr": FalsePositiveRate,
        "false_positive_rate": FalsePositiveRate,
    }

    if isinstance(metric, str):
        if metric.lower() in METRICS_ALIASES:
            metric = METRICS_ALIASES[metric.lower()](name=metric.lower())
        else:
            raise ValueError(f"Unknown metric name: {metric}, typo?")

    if name:
        metric.name = name

    return metric
