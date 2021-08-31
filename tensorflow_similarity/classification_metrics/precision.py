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

        # If all queries return empty result sets we have a recall of zero. In
        # this case the precision should be 1.0 (see
        # https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html#fig:precision-recall).
        # The following sets the first precision value to 1.0 if the first
        # recall and precision are both zero.
        if (tp + fp)[0] == 0.0 and len(p) > 1:
            initial_precsion = tf.constant(
                    [tf.constant([1.0]), tf.zeros(len(p)-1)],
                    axis=0
            )
            p = p + initial_precsion

        return p
