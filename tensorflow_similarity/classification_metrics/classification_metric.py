from abc import abstractmethod
from abc import ABC

from tensorflow_similarity.types import FloatTensor, IntTensor


class ClassificationMetric(ABC):
    """Abstract base class for computing classification metrics.

    Attributes:
        name: Name associated with the metric object, e.g., accuracy.

        canonical_name: The canonical name associated with metric,
        e.g., accuracy
    """

    def __init__(self,
                 name: str = '',
                 canonical_name: str = '',
                 direction: str = 'max') -> None:
        self.name = name
        self.canonical_name = canonical_name
        self.direction = direction

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return "%s:%s" % (self.canonical_name, self.name)

    def get_config(self):
        return {
            "name": str(self.name),
            "canonical_name": str(self.canonical_name),
        }

    @abstractmethod
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
