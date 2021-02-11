import numpy as np
from tensorflow_similarity.matchers import NMSLibMatcher


def test_index_match():
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    matcher = NMSLibMatcher()
    matcher.add(embs[0], 0)
    matcher.add(embs[1], 1)

    idxs, embs = matcher.lookup(target)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    matcher = NMSLibMatcher()
    matcher.add(embs[0], 0)
    matcher.add(embs[1], 1)

    idxs, embs = matcher.lookup(target)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]

    matcher.save(tmp_path)

    matcher2 = NMSLibMatcher()
    matcher2.load(tmp_path)

    idxs2, embs2 = matcher.lookup(target)
    assert len(embs2) == 2
    assert list(idxs2) == [0, 1]

    # add more
    matcher2.add(np.array([3.0, 3.0, 3.0]), 3)
    idxs3, embs3 = matcher2.lookup(target)
    assert len(embs3) == 3
    assert list(idxs3) == [0, 3, 1]
