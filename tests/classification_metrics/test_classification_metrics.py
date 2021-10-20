import re

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.classification_metrics import F1Score
from tensorflow_similarity.classification_metrics import FalsePositiveRate
from tensorflow_similarity.classification_metrics import NegativePredictiveValue
from tensorflow_similarity.classification_metrics import Precision
from tensorflow_similarity.classification_metrics import Recall
from tensorflow_similarity.classification_metrics import BinaryAccuracy
from tensorflow_similarity.classification_metrics import (
    make_classification_metric,
)

testdata = [
    # Perfect classification with two distance thresholds
    {
        "tp": tf.ones((2,)),
        "fp": tf.zeros((2,)),
        "tn": tf.ones((2,)),
        "fn": tf.zeros((2,)),
        "count": 2,
    },
    # Inverted classification with two distance thresholds
    {
        "tp": tf.zeros((2,)),
        "fp": tf.ones((2,)),
        "tn": tf.zeros((2,)),
        "fn": tf.ones((2,)),
        "count": 2,
    },
    # Decent classification with two distance thresholds
    {
        "tp": tf.constant([10.0, 20.0]),
        "fp": tf.constant([1.0, 2.0]),
        "tn": tf.constant([10.0, 9.0]),
        "fn": tf.constant([15.0, 5.0]),
        "count": 36,
    },
    # Single distance threshold
    {
        "tp": tf.ones((1,)),
        "fp": tf.zeros((1,)),
        "tn": tf.ones((1,)),
        "fn": tf.zeros((1,)),
        "count": 2,
    },
]

test_ids = [
    "perfect_classification",
    "inverted_classification",
    "decent_classification",
    "single_threshold",
]

f1_expected_results = [
    tf.constant(
        [1.0, 1.0]
    ),  # Perfect classification with two distance thresholds
    tf.constant(
        [0.0, 0.0]
    ),  # Inverted classification with two distance thresholds
    tf.constant(
        [0.555556, 0.851064]
    ),  # Decent classification with two distance thresholds
    tf.constant([1.0]),  # Single distance threshold
]


@pytest.mark.parametrize(
    "counts, expected", zip(testdata, f1_expected_results), ids=test_ids
)
def test_f1(counts, expected):
    f1 = F1Score()
    results = f1.compute(**counts)
    np.testing.assert_allclose(results.numpy(), expected.numpy(), rtol=1e-5)


precision_expected_results = [
    tf.constant(
        [1.0, 1.0]
    ),  # Perfect classification with two distance thresholds
    tf.constant(
        [0.0, 0.0]
    ),  # Inverted classification with two distance thresholds
    tf.constant(
        [0.909091, 0.909091]
    ),  # Decent classification with two distance thresholds
    tf.constant([1.0]),  # Single distance threshold
]


@pytest.mark.parametrize(
    "counts, expected", zip(testdata, precision_expected_results), ids=test_ids
)
def test_precision(counts, expected):
    precision = Precision()
    results = precision.compute(**counts)
    np.testing.assert_allclose(results.numpy(), expected.numpy(), rtol=1e-5)


recall_expected_results = [
    tf.constant(
        [1.0, 1.0]
    ),  # Perfect classification with two distance thresholds
    tf.constant(
        [0.0, 0.0]
    ),  # Inverted classification with two distance thresholds
    tf.constant(
        [0.4, 0.8]
    ),  # Decent classification with two distance thresholds
    tf.constant([1.0]),  # Single distance threshold
]


@pytest.mark.parametrize(
    "counts, expected", zip(testdata, recall_expected_results), ids=test_ids
)
def test_recall(counts, expected):
    recall = Recall()
    results = recall.compute(**counts)
    np.testing.assert_allclose(results.numpy(), expected.numpy(), rtol=1e-5)


binary_accuracy_expected_results = [
    tf.constant(
        [0.5, 0.5]
    ),  # Perfect classification with two distance thresholds
    tf.constant(
        [0.0, 0.0]
    ),  # Inverted classification with two distance thresholds
    tf.constant(
        [0.277778, 0.555556]
    ),  # Decent classification with two distance thresholds
    tf.constant([0.5]),  # Single distance threshold
]


@pytest.mark.parametrize(
    "counts, expected",
    zip(testdata, binary_accuracy_expected_results),
    ids=test_ids,
)
def test_binary_accuracy(counts, expected):
    acc = BinaryAccuracy()
    results = acc.compute(**counts)
    np.testing.assert_allclose(results.numpy(), expected.numpy(), rtol=1e-5)


fpr_expected_results = [
    tf.constant(
        [0.0, 0.0]
    ),  # Perfect classification with two distance thresholds
    tf.constant(
        [1.0, 1.0]
    ),  # Inverted classification with two distance thresholds
    tf.constant(
        [0.090909, 0.181818]
    ),  # Decent classification with two distance thresholds
    tf.constant([0.0]),  # Single distance threshold
]


@pytest.mark.parametrize(
    "counts, expected", zip(testdata, fpr_expected_results), ids=test_ids
)
def test_false_positive_rate(counts, expected):
    fpr = FalsePositiveRate()
    results = fpr.compute(**counts)
    np.testing.assert_allclose(results.numpy(), expected.numpy(), rtol=1e-5)


npv_expected_results = [
    tf.constant(
        [1.0, 1.0]
    ),  # Perfect classification with two distance thresholds
    tf.constant(
        [0.0, 0.0]
    ),  # Inverted classification with two distance thresholds
    tf.constant(
        [0.4, 0.642857]
    ),  # Decent classification with two distance thresholds
    tf.constant([1.0]),  # Single distance threshold
]


@pytest.mark.parametrize(
    "counts, expected", zip(testdata, npv_expected_results), ids=test_ids
)
def test_negative_predicitve_value(counts, expected):
    npv = NegativePredictiveValue()
    results = npv.compute(**counts)
    np.testing.assert_allclose(results.numpy(), expected.numpy(), rtol=1e-5)


def test_make_classification_metric():
    metric = make_classification_metric("false_positive_rate")
    assert metric.name == "false_positive_rate"


def test_make_classification_metric_name_change():
    metric = make_classification_metric("false_positive_rate", name="foo")
    assert metric.name == "foo"


def test_make_classification_metric_bad_name():
    msg = "Unknown metric name: foo, typo?"

    with pytest.raises(ValueError, match=re.escape(msg)):
        _ = make_classification_metric("foo")
