from typing import Tuple

import tensorflow as tf

from abc import abstractmethod
from abc import ABC
import math

from tensorflow_similarity.types import FloatTensor, IntTensor, BoolTensor


class ClassificationMatch(ABC):
    """Abstract base class for computing calibration metrics.

    Attributes:
        name: Name associated with the metric object, e.g., match_acc

        canonical_name: The canonical name associated with metric,
        e.g., match_accuracy

        distance_threshold: The max distance below which a nearest neighbor is
        considered a valid match.

    """

    def __init__(self,
                 name: str = '',
                 canonical_name: str = '',
                 ) -> None:
        self.name = name
        self.canonical_name = canonical_name
        self.distance_thresholds = None

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
    def compute_match_indicators(self,
                                 query_labels: IntTensor,
                                 lookup_labels: IntTensor,
                                 lookup_distances: FloatTensor
                                 ) -> Tuple[BoolTensor, BoolTensor]:
        """Compute the indicator tensor.


        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of BoolTensors:
                label_match: A len(query_labels x 1 boolean tensor. True if
                the match label == query label, False otherwise.

                dist_mask: A len(query_labels) x len(distance_thresholds)
                boolean tensor. True if the distance of the jth match <= the
                kth distance threshold.
        """

    def compile(self,
                distance_thresholds: FloatTensor = tf.constant([math.inf])):
        """Configures the distance thresholds used during matching."""
        self.distance_thresholds = tf.sort(
                tf.cast(distance_thresholds, dtype='float32')
        )

    def match(self,
              query_labels: IntTensor,
              lookup_labels: IntTensor,
              lookup_distances: FloatTensor):
        """Compares the query_label against the match label associated with the
        lookup labels.

        Note: compile must be called before calling match.

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.
        """
        label_match, dist_mask = self.compute_match_indicators(
                query_labels=query_labels,
                lookup_labels=lookup_labels,
                lookup_distances=lookup_distances
        )

        self._compute_counts(label_match, dist_mask)

    def _compute_counts(self,
                        label_match: BoolTensor,
                        dist_mask: BoolTensor):
        self._tp = tf.math.logical_and(label_match, dist_mask)
        self._tp = tf.math.count_nonzero(self._tp, axis=0)
        self._tp = tf.cast(self._tp, dtype='float')

        self._fn = tf.math.logical_and(
                label_match,
                tf.math.logical_not(dist_mask)
        )
        self._fn = tf.math.count_nonzero(self._fn, axis=0)
        self._fn = tf.cast(self._fn, dtype='float')

        self._fp = tf.math.logical_and(
                tf.math.logical_not(label_match),
                dist_mask
        )
        self._fp = tf.math.count_nonzero(self._fp, axis=0)
        self._fp = tf.cast(self._fp, dtype='float')

        self._tn = tf.math.logical_and(
                tf.math.logical_not(label_match),
                tf.math.logical_not(dist_mask)
        )
        self._tn = tf.math.count_nonzero(self._tn, axis=0)
        self._tn = tf.cast(self._tn, dtype='float')

        self._count = len(label_match)

    @property
    def tp(self):
        return self._tp

    @property
    def fp(self):
        return self._fp

    @property
    def tn(self):
        return self._tn

    @property
    def fn(self):
        return self._fn

    @property
    def count(self):
        return self._count

    @staticmethod
    def _check_shape(query_labels, lookup_labels, lookup_distances):
        ll_shape = tf.shape(lookup_labels)
        ld_shape = tf.shape(lookup_distances)

        if tf.shape(query_labels)[0] != ll_shape[0]:
            raise ValueError('Number of query labels must match the number of '
                             'lookup_label sets.')

        if ll_shape[0] != ld_shape[0] or ll_shape[1] != ld_shape[1]:
            raise ValueError('Number of number of lookup labels must match '
                             'the number of lookup distances.')
