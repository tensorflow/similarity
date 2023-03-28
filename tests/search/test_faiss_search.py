import numpy as np

from tensorflow_similarity.search import FaissSearch


def test_index_match():
    target = np.array([1, 1, 2], dtype="float32")
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target, k=2)
    print(f"idxs={idxs}, embs={embs}")

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = np.array([1, 1, 2], dtype="float32")
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")
    k = 2

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target, k=k)
    print(f"idxs={idxs}, embs={embs}")

    assert len(embs) == k
    assert list(idxs) == [0, 1]

    search_index.save(tmp_path)

    search_index2 = FaissSearch("cosine", 3, algo="flat")
    search_index2.load(tmp_path)

    idxs2, embs2 = search_index.lookup(target, k=k)
    print(f"idxs2={idxs2}, embs2={embs2}")
    assert len(embs2) == k
    assert list(idxs2) == [0, 1]

    # add more
    # if the dtype is not passed we get an incompatible type error
    search_index2.add(np.array([3.0, 3.0, 3.0], dtype="float32"), 3)
    idxs3, embs3 = search_index2.lookup(target, k=3)
    print(f"idxs3={idxs3}, embs3={embs3}")
    assert len(embs3) == 3
    assert list(idxs3) == [0, 2, 1]


def test_batch_vs_single(tmp_path):
    num_targets = 10
    index_size = 100
    vect_dim = 16

    # gen
    idxs = list(range(index_size))

    targets = np.random.random((num_targets, vect_dim)).astype("float32")
    embs = np.random.random((index_size, vect_dim)).astype("float32")

    # build search_index
    search_index = FaissSearch("cosine", vect_dim, algo="flat")
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


def test_ivfpq():
    # test ivfpq ANN indexing with 100M entries
    num_targets = 10
    index_size = 10000
    vect_dim = 16

    # gen
    idxs = np.array(list(range(index_size)))

    targets = np.random.random((num_targets, vect_dim)).astype("float32")
    embs = np.random.random((index_size, vect_dim)).astype("float32")

    search_index = FaissSearch("cosine", vect_dim, algo="ivfpq")
    assert search_index.is_built() == False
    search_index.build_index(embs)
    assert search_index.is_built() == True
    last_idx = 0
    for i in range(1000):
        idxs = np.array(list(range(last_idx, last_idx + index_size)))
        embs = np.random.random((index_size, vect_dim)).astype("float32")
        last_idx += index_size
        search_index.batch_add(embs, idxs)
    found_idxs, found_dists = search_index.batch_lookup(targets, 2)
    assert found_idxs.shape == (10, 2)


def test_reset():
    target = np.array([1, 2, 3], dtype="float32")
    embs = np.array([[3, 2, 1], [2, 3, 4]], dtype="float32")

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, out_embs = search_index.lookup(target, k=2)
    print(f"idxs={idxs}, embs={out_embs}")

    assert len(out_embs) == 2
    assert list(idxs) == [1, 0]
    search_index.reset()
    # only add one
    search_index.add(embs[0], 0)
    idxs, out_embs = search_index.lookup(target, k=2)
    print(f"idxs={idxs}, embs={out_embs}")
    assert len(out_embs) == 2
    assert list(idxs) == [0, -1]
