import tensorflow as tf

from tensorflow_similarity.types import FloatTensor, IntTensor
from .classification_metric import ClassificationMetric


class Accuracy(ClassificationMetric):

    def __init__(self, name: str = 'acc') -> None:
        super().__init__(name=name, canonical_name='accuracy')

    def compute(self,
                tp: IntTensor,
                fp: IntTensor,
                tn: IntTensor,
                fn: IntTensor,
                count: int) -> FloatTensor:
        """How many correct matches are returned for the given set pf parameters
        Probably the most important metric. Binary accuracy and match() is when
        k=1.

        num_matches / num_queries

        Accuracy is technically (TP+TN)/(TP+FP+TN+FN), but here we filter all
        queries above the distance threshold. In the case of binary matching,
        this makes all the TPs and FPs below the distance threshold and all the
        TNs and FNs above the distance threshold. As we are only concerned with
        the matches below the distance threshold, the accuracy simplifies to
        TP/(TP+FP) and is equivelent to the precision with respect to the
        unfiltered queries.  However, we also want to consider the query
        coverage at the distance threshold, i.e., the predicted positive rate
        computed as (TP+FP)/(TP+FP+TN+FN). Therefore, we can take precision *
        query_coverage to produce a measure that capture the precision scaled
        by the query coverage.  This simplifies down to the accuracy presented
        here, giving TP/(TP+FP+TN+FN).

        Args:
            tp: The count of True Positives at each distance threshold.
            fp: The count of False Positives at each distance threshold.
            tn: The count of True Negatives at each distance threshold.
            fn: The count of False Negatives at each distance threshold.
            count: The total number of queries
        """
        result: FloatTensor = tp / tf.constant([count], dtype='float')
        return result
