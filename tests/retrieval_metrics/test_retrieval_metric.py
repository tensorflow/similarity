import re

import pytest
import tensorflow as tf

from tensorflow_similarity.retrieval_metrics import RetrievalMetric
from tensorflow_similarity.types import BoolTensor, FloatTensor, IntTensor


class ConcreteRetrievalMetric(RetrievalMetric):
    def compute(
        self,
        *,  # keyword only arguments see PEP-570
        query_labels: IntTensor,
        lookup_labels: IntTensor,
        lookup_distances: FloatTensor,
        match_mask: BoolTensor,
    ) -> FloatTensor:
        return tf.constant(1.0)


def test_concrete_instance():
    rm = ConcreteRetrievalMetric(
        name="foo",
        canonical_name="bar",
        k=6,
        average="macro",
    )

    assert rm.name == "foo@6"
    # Check the name once we have updated the threshold
    rm.distance_threshold = 0.1
    assert rm.name == "foo@6 : distance_threshold@0.1"
    assert repr(rm) == "bar : foo@6 : distance_threshold@0.1"
    assert rm.canonical_name == "bar"
    assert rm.k == 6
    assert rm.distance_threshold == 0.1
    assert rm.average == "macro"

    expected_config = {
        "name": "foo@6 : distance_threshold@0.1",
        "canonical_name": "bar",
        "k": 6,
        "distance_threshold": 0.1,
    }
    assert rm.get_config() == expected_config


def test_k_greater_than_num_lookups():
    query_labels = tf.constant([1, 1])
    match_mask = tf.constant(
        [
            [True, True, False, False],
            [True, False, False, True],
        ],
        dtype=bool,
    )
    rm = ConcreteRetrievalMetric(
        name="foo",
        canonical_name="bar",
        k=5,
        average="macro",
    )

    msg = "The number of neighbors must be >= K. Number of neighbors is 4 but " "K is 5."

    with pytest.raises(ValueError, match=re.escape(msg)):
        _ = rm._check_shape(query_labels=query_labels, match_mask=match_mask)


def test_query_and_match_mask_different_dims():
    query_labels = tf.constant([1, 2, 3, 4])
    match_mask = tf.constant(
        [
            [True, True, False, False],
            [True, False, False, True],
        ],
        dtype=bool,
    )
    rm = ConcreteRetrievalMetric(
        name="foo",
        canonical_name="bar",
        k=4,
        average="macro",
    )

    msg = (
        "The number of lookup sets must equal the number of query labels. "
        "Number of lookup sets is 2 but the number of query labels is 4."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        _ = rm._check_shape(query_labels=query_labels, match_mask=match_mask)
