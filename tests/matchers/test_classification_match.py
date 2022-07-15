import re
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.matchers import ClassificationMatch
from tensorflow_similarity.types import FloatTensor, IntTensor

attributes = ("tp", "fp", "tn", "fn", "count")


class ConcreteClassificationMatch(ClassificationMatch):
    def derive_match(self, lookup_labels: IntTensor, lookup_distances: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        return lookup_labels, lookup_distances


class BadClassificationMatch(ClassificationMatch):
    "Derive match should return 2D tensors, but here we return 1D."

    def derive_match(self, lookup_labels: IntTensor, lookup_distances: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        return (
            tf.reshape(lookup_labels, (-1,)),
            tf.reshape(lookup_distances, (-1,)),
        )


def test_compile():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    # Pass distance_thresholds as a 1D tensor.
    distance_thresholds = tf.constant([1, 2, 3, 4, 5, 6, 7])
    cm.compile(distance_thresholds=distance_thresholds)

    expected_dt = tf.cast(distance_thresholds, dtype="float32")
    assert cm.name == "foo"
    assert cm.canonical_name == "bar"
    assert tf.math.reduce_all(tf.shape(cm.distance_thresholds) == tf.shape(expected_dt))
    assert tf.math.reduce_all(tf.math.equal(cm.distance_thresholds, expected_dt))


def test_compute_match_indicators():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    # Pass distance_thresholds as a 1D tensor.
    distance_thresholds = tf.constant([1.0, 2.0])
    cm.compile(distance_thresholds=distance_thresholds)

    query_labels = tf.constant([10, 20, 10, 20])
    lookup_labels = tf.constant([[10], [20], [30], [40]])
    lookup_distances = tf.constant([[1.0], [1.0], [2.0], [2.0]])

    match_mask, distance_mask = cm._compute_match_indicators(query_labels, lookup_labels, lookup_distances)

    np.testing.assert_array_equal(match_mask.numpy(), np.array([[True], [True], [False], [False]]))

    np.testing.assert_array_equal(
        distance_mask.numpy(),
        np.array([[True, True], [True, True], [False, True], [False, True]]),
    )


def test_compute_match_indicators_1d():
    """Check that we handle 1D derive match results."""
    cm = BadClassificationMatch(name="foo", canonical_name="bar")

    # Pass distance_thresholds as a 1D tensor.
    distance_thresholds = tf.constant([1.0, 2.0])
    cm.compile(distance_thresholds=distance_thresholds)

    query_labels = tf.constant([10, 20, 10, 20])
    lookup_labels = tf.constant([[10], [20], [30], [40]])
    lookup_distances = tf.constant([[1.0], [1.0], [2.0], [2.0]])

    match_mask, distance_mask = cm._compute_match_indicators(query_labels, lookup_labels, lookup_distances)

    np.testing.assert_array_equal(match_mask.numpy(), np.array([[True], [True], [False], [False]]))

    np.testing.assert_array_equal(
        distance_mask.numpy(),
        np.array([[True, True], [True, True], [False, True], [False, True]]),
    )


def test_compute_count():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    # Pass distance_thresholds as a 1D tensor.
    distance_thresholds = tf.constant([1.0, 2.0])
    cm.compile(distance_thresholds=distance_thresholds)

    query_labels = tf.constant([10, 20, 10, 20])
    lookup_labels = tf.constant([[10], [20], [30], [40]])
    lookup_distances = tf.constant([[1.0], [1.0], [2.0], [2.0]])

    cm.compute_count(query_labels, lookup_labels, lookup_distances)

    np.testing.assert_array_equal(cm.tp.numpy(), np.array([2, 2]))
    np.testing.assert_array_equal(cm.fp.numpy(), np.array([0, 2]))
    np.testing.assert_array_equal(cm.tn.numpy(), np.array([2, 0]))
    np.testing.assert_array_equal(cm.fn.numpy(), np.array([0, 0]))
    assert cm.count == 4


@pytest.mark.parametrize("attribute", attributes, ids=attributes)
def test_attribute_asserts(attribute):
    """Uninitialized attrs should through a ValueError."""
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    msg = "Matcher.compute_count() must be called before accessing the counts."

    with pytest.raises(AttributeError, match=re.escape(msg)):
        getattr(cm, attribute)


def test_check_shape_valid():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    queries = tf.constant([[1], [2], [3]])
    ll = tf.constant([[1], [2], [3]])
    ld = tf.constant([[0.1], [0.2], [0.3]])

    assert cm._check_shape(queries, ll, ld)


def test_check_shape_invalid_queries():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    queries = tf.constant([[1], [2], [3], [4]])
    ll = tf.constant([[1], [2], [3]])
    ld = tf.constant([[0.1], [0.2], [0.3]])

    msg = "Number of query labels must match the number of lookup_label sets."

    with pytest.raises(ValueError, match=re.escape(msg)):
        cm._check_shape(queries, ll, ld)


def test_check_shape_invalid_lookup_rank():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    queries = tf.constant([[1], [2], [3]])
    ll = tf.constant([1, 2, 3])
    ld = tf.constant([[0.1], [0.2], [0.3]])

    msg = "lookup_labels must be a 2D tensor of shape [len(query_labels), K]."

    with pytest.raises(ValueError, match=re.escape(msg)):
        cm._check_shape(queries, ll, ld)


def test_check_shape_invalid_distance_rank():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    queries = tf.constant([[1], [2], [3]])
    ll = tf.constant([[1], [2], [3]])
    ld = tf.constant([0.1, 0.2, 0.3])

    msg = "lookup_distances must be a 2D tensor of shape " "[len(query_labels), K]."

    with pytest.raises(ValueError, match=re.escape(msg)):
        cm._check_shape(queries, ll, ld)


def test_check_shape_labels_dist_mismatch():
    cm = ConcreteClassificationMatch(name="foo", canonical_name="bar")

    queries = tf.constant([[1], [2], [3]])
    ll = tf.constant([[1], [2], [3]])
    ld = tf.constant([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])

    msg = "Number of number of lookup labels must match the number " "of lookup distances."

    with pytest.raises(ValueError, match=re.escape(msg)):
        cm._check_shape(queries, ll, ld)
