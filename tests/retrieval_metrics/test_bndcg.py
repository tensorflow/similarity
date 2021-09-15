import math
import re

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.retrieval_metrics import BNDCG


@pytest.fixture
def test_data():
    return {
        "query_labels": tf.constant([1, 1, 1, 0]),
        "lookup_distances": tf.constant(
            [
                [0.0, 0.1, 0.2],
                [0.0, 0.1, 0.2],
                [0.0, 0.1, 0.2],
                [0.0, 0.1, 0.2],
            ],
            dtype=float,
        ),
        "match_mask": tf.constant(
            [
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ],
            dtype=bool,
        ),
    }


def test_compute_distance_threshold(test_data):
    """Test filtering using distance threshold."""
    rm = BNDCG(k=3, distance_threshold=0.1)

    bndcg = rm.compute(
        query_labels=test_data["query_labels"],
        lookup_distances=test_data["lookup_distances"],
        match_mask=test_data["match_mask"],
    )

    expected = tf.constant(0.75)

    np.testing.assert_allclose(bndcg, expected)


def test_compute_at_k(test_data):
    """Test filtering using K."""
    rm = BNDCG(k=2)

    assert rm.distance_threshold == math.inf

    bndcg = rm.compute(
        query_labels=test_data["query_labels"],
        lookup_distances=test_data["lookup_distances"],
        match_mask=test_data["match_mask"],
    )

    expected = tf.constant(0.75)

    np.testing.assert_allclose(bndcg, expected)


def test_compute_macro(test_data):
    """Test filtering using K."""
    rm = BNDCG(k=2, average="macro")

    assert rm.distance_threshold == math.inf

    bndcg = rm.compute(
        query_labels=test_data["query_labels"],
        lookup_distances=test_data["lookup_distances"],
        match_mask=test_data["match_mask"],
    )

    expected = tf.constant(0.5)

    np.testing.assert_allclose(bndcg, expected)


def test_query_and_lookup_distances_different_dims():
    query_labels = tf.constant([1, 2, 3, 4])
    lookup_distances = tf.constant(
        [
            [0.0, 0.1, 0.2],
            [0.0, 0.1, 0.2],
        ],
        dtype=float,
    )
    match_mask = tf.constant(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )

    rm = BNDCG(k=3)

    msg = (
        "The number of lookup distance rows must equal the number of query "
        "labels. Number of lookup distance rows is 2 but the number of "
        "query labels is 4."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        _ = rm.compute(
            query_labels=query_labels,
            lookup_distances=lookup_distances,
            match_mask=match_mask,
        )
