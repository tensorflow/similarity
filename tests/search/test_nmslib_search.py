import numpy as np
from tensorflow_similarity.search import NMSLibSearch


def test_index_match():
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    search_index = NMSLibSearch('cosine', 3)
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    search_index = NMSLibSearch('cosine', 3)
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]

    search_index.save(tmp_path)

    search_index2 = NMSLibSearch('cosine', 3)
    search_index2.load(tmp_path)

    idxs2, embs2 = search_index.lookup(target)
    assert len(embs2) == 2
    assert list(idxs2) == [0, 1]

    # add more
    search_index2.add(np.array([3.0, 3.0, 3.0]), 3)
    idxs3, embs3 = search_index2.lookup(target)
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

    # build search_index
    search_index = NMSLibSearch('cosine', vect_dim)
    search_index.batch_add(embs, idxs)

    # batch
    batch_idxs, _ = search_index.batch_lookup(targets)

    # single
    singles_idxs = []
    for t in targets:
        idxs, embs = search_index.lookup(t)
        singles_idxs.append(idxs)

    for i in range(num_targets):
        # k neigboors are the same?
        for k in range(3):
            assert batch_idxs[i][k] == singles_idxs[i][k]
