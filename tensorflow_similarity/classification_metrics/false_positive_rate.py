import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, IntTensor
from .classification_metric import ClassificationMetric


class FalsePositiveRate(ClassificationMetric):

    def __init__(self, name: str = 'fpr') -> None:
        super().__init__(name=name, canonical_name='false_positive_rate')

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
        return tf.math.divide_no_nan(fp, fp + tn)
