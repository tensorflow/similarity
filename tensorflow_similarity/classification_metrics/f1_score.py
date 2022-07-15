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


class F1Score(ClassificationMetric):
    r"""Calculates the harmonic mean of precision and recall.

    Computes the F-1 Score given the query classification counts. The metric is
    computed as follows:

    $$
    F_1 = 2 \cdot \frac{\textrm{precision} \cdot \textrm{recall}}{\textrm{precision} + \textrm{recall}}
    $$

    args:
        name: Name associated with a specific metric object, e.g.,
        f1@0.1

    Usage with `tf.similarity.models.SimilarityModel()`:

    ```python
    model.calibrate(x=query_examples,
                    y=query_labels,
                    calibration_metric='f1')
    ```
    """

    def __init__(self, name: str = "f1") -> None:
        super().__init__(name=name, canonical_name="f1_score")

    def compute(
        self,
        tp: FloatTensor,
        fp: FloatTensor,
        tn: FloatTensor,
        fn: FloatTensor,
        count: int,
    ) -> FloatTensor:
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
        recall = tf.math.divide_no_nan(tp, tp + fn)
        precision = tf.math.divide_no_nan(tp, tp + fp)
        numer = 2 * recall * precision
        denom = recall + precision
        result: FloatTensor = tf.math.divide_no_nan(numer, denom)
        return result
