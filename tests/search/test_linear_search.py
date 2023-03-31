import numpy as np

from tensorflow_similarity.search import LinearSearch


def test_index_match():
    target = np.array([1, 1, 2], dtype="float32")
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

    search_index = LinearSearch("cosine", 3)
    search_index.add(embs[0], 0, normalize=False)
    search_index.add(embs[1], 1, normalize=False)

    idxs, embs = search_index.lookup(target, k=2, normalize=False)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_reset():
    target = np.array([1, 2, 3], dtype="float32")
    embs = np.array([[4, 2, 1], [2, 3, 5]], dtype="float32")

    search_index = LinearSearch("cosine", 3)
    search_index.add(embs[0], 0, normalize=True)
    search_index.add(embs[1], 1, normalize=True)

    idxs, dists = search_index.lookup(target, k=2, normalize=True)

    assert len(dists) == 2
    print(dists)
    assert list(idxs) == [1, 0]

    search_index.reset()
    # switch order
    search_index.add(embs[1], 0, normalize=True)
    search_index.add(embs[0], 1, normalize=True)
    idxs, dists = search_index.lookup(target, k=2, normalize=True)

    assert len(dists) == 2
    assert list(idxs) == [0, 1]


def test_index_match_l1():
    target = np.array([1, 1, 2], dtype="float32")
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

    search_index = LinearSearch("l1", 3)
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target, k=2)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_match_l2():
    target = np.array([1, 1, 2], dtype="float32")
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

    search_index = LinearSearch("l2", 3)
    search_index.add(embs[0], 0, normalize=False)
    search_index.add(embs[1], 1, normalize=False)

    idxs, embs = search_index.lookup(target, k=2, normalize=False)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = np.array([1, 1, 2], dtype="float32")
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")
    k = 2

    search_index = LinearSearch("cosine", 3)
    search_index.add(embs[0], 0, normalize=False)
    search_index.add(embs[1], 1, normalize=False)

    idxs, embs = search_index.lookup(target, k=k, normalize=False)

    assert len(embs) == k
    assert list(idxs) == [0, 1]

    search_index.save(tmp_path)

    search_index2 = LinearSearch("cosine", 3)
    search_index2.load(tmp_path)

    idxs2, embs2 = search_index.lookup(target, k=k, normalize=False)
    assert len(embs2) == k
    assert list(idxs2) == [0, 1]

    # add more
    # if the dtype is not passed we get an incompatible type error
    search_index2.add(np.array([3.0, 3.0, 3.0], dtype="float32"), 3, normalize=False)
    idxs3, embs3 = search_index2.lookup(target, k=3, normalize=False)
    assert len(embs3) == 3
    assert list(idxs3) == [0, 1, 3]


def test_batch_vs_single(tmp_path):
    num_targets = 10
    index_size = 100
    vect_dim = 16

    # gen
    idxs = list(range(index_size))

    targets = np.random.random((num_targets, vect_dim)).astype("float32")
    embs = np.random.random((index_size, vect_dim)).astype("float32")

    # build search_index
    search_index = LinearSearch("cosine", vect_dim)
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


def test_running_larger_batches():
    num_targets = 10
    index_size = 300
    vect_dim = 16

    # gen
    idxs = np.array(list(range(index_size)))

    targets = np.random.random((num_targets, vect_dim)).astype("float32")
    embs = np.random.random((index_size, vect_dim)).astype("float32")

    search_index = LinearSearch("cosine", vect_dim)
    assert search_index.is_built() == True
    last_idx = 0
    for i in range(1000):
        idxs = np.array(list(range(last_idx, last_idx + index_size)))
        embs = np.random.random((index_size, vect_dim)).astype("float32")
        last_idx += index_size
        search_index.batch_add(embs, idxs)
    found_idxs, found_dists = search_index.batch_lookup(targets, 2)
    assert len(found_idxs) == 10
    assert len(found_idxs[0]) == 2
