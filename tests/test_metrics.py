from tensorflow_similarity.metrics import MinRank, MeanRank, MaxRank
from tensorflow_similarity.metrics import Accuracy, Precision, Recall, F1Score

MAX_K = 0
TARGETS_LABELS = 1
NUM_MATCHED = 2
NUM_UNMATCHED = 3
INDEX_SIZE = 4
MATCH_RANKS = 5
MATCH_DISTANCES = 6
MATCH_LABELS = 7

TEST_VECTORS = [[
    3,  # max_k
    [1, 2, 3, 4],  # targets_labels
    3,  # num_matched
    1,  # num_unmatched
    30,  # index size
    [2, 1, 2, 0],  # match_ranks
    [0.1, 0.2, 0.3, 0],  # match_distances
    [  # lookups
        [{
            'label': 21,
            'distance': 0.01
        }, {
            'label': 1,
            'distance': 0.1
        }],
        [{
            'label': 2,
            'distance': 0.2
        }, {
            'label': 22,
            'distance': 0.22
        }],
        [{
            'label': 23,
            'distance': 0.01
        }, {
            'label': 3,
            'distance': 0.3
        }],
        [{
            'label': 24,
            'distance': 0.44
        }, {
            'label': 44,
            'distance': 0.54
        }]
    ]
]]


def compute_vector(m, v):
    return m.compute(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])


def test_mean_rank():
    expected = [1]
    m = MeanRank()

    # check name
    assert m.name == 'mean_rank'

    # check computation
    for idx, v in enumerate(TEST_VECTORS):
        res = compute_vector(m, v)
        assert res == expected[idx]


def test_min_rank():
    expected = [1]
    m = MinRank()

    # check name
    assert m.name == 'min_rank'

    # check computation
    for idx, v in enumerate(TEST_VECTORS):
        res = compute_vector(m, v)
        assert res == expected[idx]


def test_max_rank():
    expected = [2]
    m = MaxRank()

    # check name
    assert m.name == 'max_rank'

    # check computation
    for idx, v in enumerate(TEST_VECTORS):
        res = compute_vector(m, v)
        assert res == expected[idx]


def test_f1_score():
    expected = [3]
    m = F1Score()

    # check name
    assert m.name == 'f1_score'

    # check computation
    for idx, v in enumerate(TEST_VECTORS):
        res = compute_vector(m, v)
        # FIXME compute the real value and check for it
        # assert res == expected[idx]


# lower level functions
def test_filter_rank_k():
    match_ranks = [1, 2, 3, 1, 2, 3]
    match_distances = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    m = MinRank()
    for k in range(1, 3):
        matches = m.filter_ranks(match_ranks,
                                 match_distances,
                                 max_rank=k,
                                 distance=1)
        assert min(matches) == 1
        assert max(matches) == k


def test_filter_rank_distance():
    match_ranks = [1, 2, 3, 1, 2, 3]
    match_distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    m = MinRank()
    for i in range(1, 7):
        distance = i / 10
        matches = m.filter_ranks(match_ranks,
                                 match_distances,
                                 max_rank=10,
                                 distance=distance)
        assert len(matches) == i
