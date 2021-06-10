from typing import Tuple
import tensorflow as tf
from tensorflow_similarity.metrics import batch_class_ratio
from tensorflow_similarity.metrics import MaxRank
from tensorflow_similarity.metrics import MeanRank
from tensorflow_similarity.metrics import MinRank
from tensorflow_similarity.samplers.samplers import Sampler
from tensorflow_similarity.types import Lookup, Tensor


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
    30,  # index size
    [2, 1, 2, 0],  # match_ranks
    [0.1, 0.2, 0.3, 0],  # match_distances
    [  # lookups
        [
            Lookup(rank=0, label=21, distance=0.01),
            Lookup(rank=1, label=1, distance=0.1)
         ],
        [
            Lookup(rank=0, label=2, distance=0.2),
            Lookup(rank=1, label=22, distance=0.22)
         ],
        [
            Lookup(rank=0, label=23, distance=0.01),
            Lookup(rank=1, label=3, distance=0.3)
         ],
        [
            Lookup(rank=0, label=24, distance=0.44),
            Lookup(rank=1, label=44, distance=0.54)
         ]
    ]
]]


def compute_vector(m, v):
    return m.compute(v[0], v[1], v[2], v[3], v[4], v[5])


def test_mean_rank():
    expected = [1.67]
    m = MeanRank()

    # check name
    assert m.name == 'mean_rank'

    # check computation
    for idx, v in enumerate(TEST_VECTORS):
        res = compute_vector(m, v)
        assert round(res, 2) == expected[idx]


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
    # expected = [3]
    # m = F1Score()

    # # check name
    # assert m.name == 'f1_score'

    # # check computation
    # for idx, v in enumerate(TEST_VECTORS):
    #     res = compute_vector(m, v)
    #     # FIXME compute the real value and check for it
    #     # assert res == expected[idx]
    pass


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


def test_batch_class_ratio():
    class MockSampler(Sampler):

        def __init__(self):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, *args) -> Tuple[Tensor, Tensor]:
            return (tf.ones(8),  tf.constant([0, 0, 1, 1, 2, 2, 3, 3]))

        def get_examples(self):
            pass

    result = batch_class_ratio(MockSampler())
    assert result == 2.0
