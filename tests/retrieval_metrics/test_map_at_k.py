import numpy as np
import tensorflow as tf

from tensorflow_similarity.retrieval_metrics import MapAtK


def test_concrete_instance():
    rm = MapAtK(r={1: 4, 0: 4})

    assert rm.name == "map@5"
    # Check the name once we have updated the threshold
    rm.distance_threshold = 0.1
    assert rm.name == "map@5 : distance_threshold@0.1"
    assert repr(rm) == "map@k : map@5 : distance_threshold@0.1"
    assert rm.canonical_name == "map@k"
    assert rm.k == 5
    assert rm.distance_threshold == 0.1
    assert rm.average == "micro"

    expected_config = {
        "r": {1: 4, 0: 4},
        "name": "map@5 : distance_threshold@0.1",
        "canonical_name": "map@k",
        "k": 5,
        "distance_threshold": 0.1,
    }
    assert rm.get_config() == expected_config


def test_compute():
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
    rm = MapAtK(r={0: 10, 1: 3},  k=3)

    mapk = rm.compute(query_labels=query_labels, match_mask=match_mask)

    # mapk should be sum(precision@k*Relevancy_Mask)/R
    # class 1 has 3 results sets that are all T,T,F:
    #     (1.0*True+1.0*True+0.66*False)/3 = 0.66667
    # class 0 has 1 result set that is F,F,T
    #     (0.0*False+0.0*False+0.33*True)/10 = 0.03332
    # mapk = (0.667*3 + 0.0332)/4
    expected = tf.constant(0.50833333332)

    np.testing.assert_allclose(mapk, expected)
