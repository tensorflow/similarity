import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, IntTensor
from .classification_metric import ClassificationMetric


class Precision(ClassificationMetric):

    def __init__(self, name: str = 'precision') -> None:
        super().__init__(name=name, canonical_name='classification_precision')

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
        p: FloatTensor = tf.math.divide_no_nan(tp, tp + fp)
        # Handle the case where we have no valid matches at a recall of 0.0
        if p[0] == 0.0 and len(p) > 1:
            p = p + tf.constant([1.0]+[0.0]*(len(p)-1))
        return p
