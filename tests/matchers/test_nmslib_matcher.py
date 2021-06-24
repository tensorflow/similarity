import numpy as np
from tensorflow_similarity.matchers import NMSLibMatcher


def test_index_match():
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    matcher = NMSLibMatcher('cosine', 3)
    matcher.add(embs[0], 0)
    matcher.add(embs[1], 1)

    idxs, embs = matcher.lookup(target)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    matcher = NMSLibMatcher('cosine', 3)
    matcher.add(embs[0], 0)
    matcher.add(embs[1], 1)

    idxs, embs = matcher.lookup(target)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]

    matcher.save(tmp_path)

    matcher2 = NMSLibMatcher('cosine', 3)
    matcher2.load(tmp_path)

    idxs2, embs2 = matcher.lookup(target)
    assert len(embs2) == 2
    assert list(idxs2) == [0, 1]

    # add more
    matcher2.add(np.array([3.0, 3.0, 3.0]), 3)
    idxs3, embs3 = matcher2.lookup(target)
    assert len(embs3) == 3
    assert list(idxs3) == [0, 3, 1]


def test_batch_vs_single(tmp_path):
    num_targets = 10
    index_size = 100
    vect_dim = 16

    # gen
    idxs = [i for i in range(index_size)]

    targets = np.random.random((num_targets, vect_dim)).astype('float32')
    embs = np.random.random((index_size, vect_dim)).astype('float32')

    # build matcher
    matcher = NMSLibMatcher('cosine', vect_dim)
    matcher.batch_add(embs, idxs)

    # batch
    batch_idxs, _ = matcher.batch_lookup(targets)

    # single
    singles_idxs = []
    for t in targets:
        idxs, embs = matcher.lookup(t)
        singles_idxs.append(idxs)

    for i in range(num_targets):
        # k neigboors are the same?
        for k in range(3):
            assert batch_idxs[i][k] == singles_idxs[i][k]
