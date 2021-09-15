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

import tensorflow as tf

from tensorflow_similarity.types import FloatTensor
from .classification_metric import ClassificationMetric


class BinaryAccuracy(ClassificationMetric):
    """Calculates how often the query label matches the derived lookup label.

    Accuracy is technically (TP+TN)/(TP+FP+TN+FN), but here we filter all
    queries above the distance threshold. In the case of binary matching, this
    makes all the TPs and FPs below the distance threshold and all the TNs and
    FNs above the distance threshold.

    As we are only concerned with the matches below the distance threshold, the
    accuracy simplifies to TP/(TP+FP) and is equivalent to the precision with
    respect to the unfiltered queries. However, we also want to consider the
    query coverage at the distance threshold, i.e., the percentage of queries
    that retrun a match, computed as (TP+FP)/(TP+FP+TN+FN). Therefore, we can
    take $ precision \times query_coverage $ to produce a measure that capture
    the precision scaled by the query coverage. This simplifies down to the
    binary accuracy presented here, giving TP/(TP+FP+TN+FN).

    args:
        name: Name associated with a specific metric object, e.g.,
        binary_accuracy@0.1

    Usage with `tf.similarity.models.SimilarityModel()`:

    ```python
    model.calibrate(x=query_examples,
                    y=query_labels,
                    calibration_metric='binary_accuracy')
    ```
    """
    def __init__(self, name: str = 'binary_accuracy') -> None:
        super().__init__(name=name, canonical_name='binary_accuracy')

    def compute(self,
                tp: FloatTensor,
                fp: FloatTensor,
                tn: FloatTensor,
                fn: FloatTensor,
                count: int) -> FloatTensor:
        """Compute the classification metric.

        The `compute()` method supports computing the metric for a set of
        values, where each value represents the counts at a specific distance
        threshold.

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
        result: FloatTensor = tp / tf.constant([count], dtype='float')
        return result
