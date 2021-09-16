import pytest
import tensorflow as tf

from tensorflow_similarity.retrieval_metrics import RecallAtK

testdata = [
    (
        "micro",
        tf.constant(0.75),
    ),
    (
        "macro",
        tf.constant(0.5),
    ),
]


@pytest.mark.parametrize("avg, expected", testdata, ids=["micro", "macro"])
def test_compute(avg, expected):
    query_labels = tf.constant([1, 1, 1, 0])
    match_mask = tf.constant(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )
    rm = RecallAtK(k=2, average=avg)

    recall = rm.compute(query_labels=query_labels, match_mask=match_mask)
    assert recall == expected
