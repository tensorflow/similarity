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
        _tp = tf.math.logical_and(label_match, dist_mask)
        _tp = tf.math.count_nonzero(_tp, axis=0)
        self._tp: FloatTensor = tf.cast(_tp, dtype='float')

        _fn = tf.math.logical_and(
                label_match,
                tf.math.logical_not(dist_mask)
        )
        _fn = tf.math.count_nonzero(_fn, axis=0)
        self._fn: FloatTensor = tf.cast(_fn, dtype='float')

        _fp = tf.math.logical_and(
                tf.math.logical_not(label_match),
                dist_mask
        )
        _fp = tf.math.count_nonzero(_fp, axis=0)
        self._fp: FloatTensor = tf.cast(_fp, dtype='float')

        _tn = tf.math.logical_and(
                tf.math.logical_not(label_match),
                tf.math.logical_not(dist_mask)
        )
        _tn = tf.math.count_nonzero(_tn, axis=0)
        self._tn: FloatTensor = tf.cast(_tn, dtype='float')

        self._count = len(label_match)

    @property
    def tp(self) -> FloatTensor:
        """The count of True positive matches.

        A True positive match is when the query label == the label generated by
        a matcher and the distance to the query is less than the distance
        threshold.

        Raises:
            AttributeError: Matcher.compute() must be called before accessing
            counts.
        """
        try:
            return self._tp
        except AttributeError as attribute_error:
            raise AttributeError(f'Matcher.compute() must be called before '
                                 'accessing the counts.') from attribute_error


    @property
    def fp(self) -> FloatTensor:
        """The count of False positive matches.

        A False positive match is when the query label != the label generated
        by a matcher and the distance to the query is less than the distance
        threshold.

        Raises:
            AttributeError: Matcher.match() must be called before accessing
            counts.
        """
        try:
            return self._fp
        except AttributeError as attribute_error:
            raise AttributeError(f'Matcher.match() must be called before '
                                 'accessing the counts.') from attribute_error

    @property
    def tn(self) -> FloatTensor:
        """The count of True negatives matches.

        A True negative match is when the query label != the label generated
        by a matcher and the distance to the query is greater than the distance
        threshold.

        Raises:
            AttributeError: Matcher.match() must be called before accessing
            counts.
        """
        try:
            return self._tn
        except AttributeError as attribute_error:
            raise AttributeError(f'Matcher.match() must be called before '
                                 'accessing the counts.') from attribute_error

    @property
    def fn(self) -> FloatTensor:
        """The count of False negatives matches.

        A False negative match is when the query label == the label generated
        by a matcher and the distance to the query is greater than the distance
        threshold.

        Raises:
            AttributeError: Matcher.match() must be called before accessing
            counts.
        """
        try:
            return self._fn
        except AttributeError as attribute_error:
            raise AttributeError(f'Matcher.match() must be called before '
                                 'accessing the counts.') from attribute_error

    @property
    def count(self):
        """The total number of queries.

        Raises:
            AttributeError: Matcher.match() must be called before accessing
            counts.
        """
        try:
            return self._count
        except AttributeError as attribute_error:
            raise AttributeError(f'Matcher.match() must be called before '
                                 'accessing the counts.') from attribute_error

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
