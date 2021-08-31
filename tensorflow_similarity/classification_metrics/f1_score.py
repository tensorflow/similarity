import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, IntTensor
from .classification_metric import ClassificationMetric


class F1Score(ClassificationMetric):

    def __init__(self, name: str = 'f1') -> None:
        super().__init__(
                name=name,
                canonical_name='f1_score')

    def compute(self,
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
        recall = tf.math.divide_no_nan(tp, tp + fn)
        precision = tf.math.divide_no_nan(tp, tp + fp)
        numer = 2 * recall * precision
        denom = recall + precision
        result: FloatTensor = tf.math.divide_no_nan(numer, denom)
        return result
