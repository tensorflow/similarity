import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, IntTensor
from .classification_metric import ClassificationMetric


class NegativePredictiveValue(ClassificationMetric):

    def __init__(self, name: str = 'npv') -> None:
        super().__init__(name=name, canonical_name='neg_predictive_value')

    def compute(self,
                *,
                tp: IntTensor,
                fp: IntTensor,
                tn: IntTensor,
                fn: IntTensor,
                count: int) -> FloatTensor:
        """Compute the classification metric.

        Args:
            tp: The count of True Positives at each distance threshold.
            fp: The count of False Positives at each distance threshold.
            tn: The count of True Negatives at each distance threshold.
            fn: The count of False Negatives at each distance threshold.
            count: The total number of queries
        """
        return tf.math.divide_no_nan(tn, tn + fn)
