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

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import tensorflow as tf

from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor


class ClassificationMatch(ABC):
    """Abstract base class for defining the classification matching strategy.

    Attributes:
        name: Name associated with the match object, e.g., match_nearest

        canonical_name: The canonical name associated with match strategy,
        e.g., match_nearest

        distance_thresholds: The max distance below which a nearest neighbor is
        considered a valid match. Defaults to None and must be set using
        `ClassificationMatch.compile()`.
    """

    def __init__(self, name: str = "", canonical_name: str = "") -> None:
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
            "distance_thresholds": self.distance_thresholds,
        }

    def compile(self, distance_thresholds: Optional[FloatTensor] = None):
        """Configures the distance thresholds used during matching.

        Args:
            distance_thresholds: The max distance below which a nearest neighbor
            is considered a valid match. A threshold of math.inf is used if None
            is passed.
        """
        if distance_thresholds is None:
            distance_thresholds = tf.constant([math.inf])

        self.distance_thresholds = tf.sort(tf.cast(distance_thresholds, dtype="float32"))

    @abstractmethod
    def derive_match(self, lookup_labels: IntTensor, lookup_distances: FloatTensor) -> Tuple[IntTensor, FloatTensor]:
        """Derive a match label and distance from a set of K neighbors.

        For each query, derive a single match label and distance given the
        associated set of lookup labels and distances.

        Args:
            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of FloatTensors:
                derived_labels: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the derived
                label for the jth query.

                derived_distances: A FloatTensor of shape
                [len(lookup_labels), 1] where the jth row contains the distance
                associated with the jth derived label.
        """

    def _compute_match_indicators(
        self, query_labels: IntTensor, lookup_labels: IntTensor, lookup_distances: FloatTensor
    ) -> Tuple[BoolTensor, BoolTensor]:
        """Compute the match indicator tensor.

        Compute the match indicator tensor given a set of query labels and a
        set of associated lookup labels and distances.

        The method first calls `derive_match()` on the lookup labels and
        distances and then compares the query labels (y_true) against the
        derived labels (y_pred) and checks if the derived distance is <= to the
        distance thresholds.

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.

        Returns:
            A Tuple of BoolTensors:
                match_mask: A BoolTensor of shape [len(query_labels), 1]. True
                if the match label == query label, False otherwise.

                distance_mask: A BoolTensor of shape [len(query_labels),
                len(distance_thresholds)]. True if the distance of the jth
                match <= the kth distance threshold.
        """
        if tf.rank(query_labels) == 1:
            query_labels = tf.expand_dims(query_labels, axis=-1)

        ClassificationMatch._check_shape(query_labels, lookup_labels, lookup_distances)

        d_labels, d_dist = self.derive_match(lookup_labels, lookup_distances)

        # Safety check to ensure that self.derive_match returns the right
        # shapes.
        if tf.rank(d_labels) == 1:
            d_labels = tf.expand_dims(d_labels, axis=-1)
        if tf.rank(d_dist) == 1:
            d_dist = tf.expand_dims(d_dist, axis=-1)

        # A 1D BoolTensor [len(query_labels), 1]
        match_mask = tf.math.equal(d_labels, query_labels)

        # A 2D BoolTensor [len(lookup_distance), len(self.distance_thresholds)]
        distance_mask = tf.math.less_equal(d_dist, self.distance_thresholds)

        return match_mask, distance_mask

    def compute_count(self, query_labels: IntTensor, lookup_labels: IntTensor, lookup_distances: FloatTensor) -> None:
        """Computes the match counts at each of the distance thresholds.

        This method computes the following at each distance threshold.

        * True Positive: The query label matches the derived lookup label and
        the derived lookup distance is <= the current distance threshold.

        * False Positive: The query label does not match the derived lookup
        label but the derived lookup distance is <= the current distance
        threshold.

        * False Negative: The query label matches the derived lookup label but
        the derived lookup distance is > the current distance threshold.

        * True Negative: The query label does not match the derived lookup
        label and the derived lookup distance is > the current distance
        threshold.

        Note: compile must be called before calling match.

        Args:
            query_labels: A 1D array of the labels associated with the queries.

            lookup_labels: A 2D array where the jth row is the labels
            associated with the set of k neighbors for the jth query.

            lookup_distances: A 2D array where the jth row is the distances
            between the jth query and the set of k neighbors.
        """
        match_mask, distance_mask = self._compute_match_indicators(
            query_labels=query_labels, lookup_labels=lookup_labels, lookup_distances=lookup_distances
        )

        self._compute_count(match_mask, distance_mask)

    def _compute_count(self, label_match: BoolTensor, dist_mask: BoolTensor) -> None:
        _tp = tf.math.logical_and(label_match, dist_mask)
        _tp = tf.math.count_nonzero(_tp, axis=0)
        self._tp: FloatTensor = tf.cast(_tp, dtype="float")

        _fn = tf.math.logical_and(label_match, tf.math.logical_not(dist_mask))
        _fn = tf.math.count_nonzero(_fn, axis=0)
        self._fn: FloatTensor = tf.cast(_fn, dtype="float")

        _fp = tf.math.logical_and(tf.math.logical_not(label_match), dist_mask)
        _fp = tf.math.count_nonzero(_fp, axis=0)
        self._fp: FloatTensor = tf.cast(_fp, dtype="float")

        _tn = tf.math.logical_and(tf.math.logical_not(label_match), tf.math.logical_not(dist_mask))
        _tn = tf.math.count_nonzero(_tn, axis=0)
        self._tn: FloatTensor = tf.cast(_tn, dtype="float")

        self._count = len(label_match)

    @property
    def tp(self) -> FloatTensor:
        """The count of True positive matches.

        A True positive match is when the query label == the label generated by
        a matcher and the distance to the query is less than the distance
        threshold.

        Raises:
            AttributeError: Matcher.compute_count() must be called before
            accessing counts.
        """
        try:
            return self._tp
        except AttributeError as attribute_error:
            raise AttributeError(
                "Matcher.compute_count() must be called before " "accessing the counts."
            ) from attribute_error

    @property
    def fp(self) -> FloatTensor:
        """The count of False positive matches.

        A False positive match is when the query label != the label generated
        by a matcher and the distance to the query is less than the distance
        threshold.

        Raises:
            AttributeError: Matcher.compute_count() must be called before
            accessing counts.
        """
        try:
            return self._fp
        except AttributeError as attribute_error:
            raise AttributeError(
                "Matcher.compute_count() must be called before " "accessing the counts."
            ) from attribute_error

    @property
    def tn(self) -> FloatTensor:
        """The count of True negatives matches.

        A True negative match is when the query label != the label generated
        by a matcher and the distance to the query is greater than the distance
        threshold.

        Raises:
            AttributeError: Matcher.compute_count() must be called before
            accessing counts.
        """
        try:
            return self._tn
        except AttributeError as attribute_error:
            raise AttributeError(
                "Matcher.compute_count() must be called before " "accessing the counts."
            ) from attribute_error

    @property
    def fn(self) -> FloatTensor:
        """The count of False negatives matches.

        A False negative match is when the query label == the label generated
        by a matcher and the distance to the query is greater than the distance
        threshold.

        Raises:
            AttributeError: Matcher.compute_count() must be called before
            accessing counts.
        """
        try:
            return self._fn
        except AttributeError as attribute_error:
            raise AttributeError(
                "Matcher.compute_count() must be called before " "accessing the counts."
            ) from attribute_error

    @property
    def count(self) -> int:
        """The total number of queries.

        Raises:
            AttributeError: Matcher.compute_count() must be called before
            accessing counts.
        """
        try:
            return self._count
        except AttributeError as attribute_error:
            raise AttributeError(
                "Matcher.compute_count() must be called before " "accessing the counts."
            ) from attribute_error

    @staticmethod
    def _check_shape(query_labels, lookup_labels, lookup_distances) -> bool:
        if tf.rank(lookup_labels) != 2:
            raise ValueError("lookup_labels must be a 2D tensor of " "shape [len(query_labels), K].")
        if tf.rank(lookup_distances) != 2:
            raise ValueError("lookup_distances must be a 2D tensor of " "shape [len(query_labels), K].")

        q_shape = tf.shape(query_labels)
        ll_shape = tf.shape(lookup_labels)
        ld_shape = tf.shape(lookup_distances)

        if q_shape[0] != ll_shape[0]:
            raise ValueError("Number of query labels must match the number of " "lookup_label sets.")

        if ll_shape[0] != ld_shape[0] or ll_shape[1] != ld_shape[1]:
            raise ValueError("Number of number of lookup labels must match " "the number of lookup distances.")

        return True
