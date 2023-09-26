import numpy as np
import tensorflow as tf

from tensorflow_similarity.search.faiss import FaissSearch


def test_index_match():
    target = tf.math.l2_normalize(np.array([[1, 1, 2]], dtype="float32"), axis=-1)
    target = np.array(target)[0]
    embs = tf.math.l2_normalize(np.array([[1, 1, 3], [3, 1, 2]], dtype="float32"), axis=-1)
    embs = np.array(embs)

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target, k=2)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = tf.math.l2_normalize(np.array([[1, 1, 2]], dtype="float32"), axis=-1)
    target = np.array(target)[0]
    embs = tf.math.l2_normalize(np.array([[1, 1, 3], [3, 1, 2]], dtype="float32"), axis=-1)
    embs = np.array(embs)
    k = 2

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target, k=k)

    assert len(embs) == k
    assert list(idxs) == [0, 1]

    search_index.save(tmp_path)

    search_index2 = FaissSearch("cosine", 3, algo="flat")
    search_index2.load(tmp_path)

    idxs2, embs2 = search_index.lookup(target, k=k)

    assert len(embs2) == k
    assert list(idxs2) == [0, 1]

    # add more
    # if the dtype is not passed we get an incompatible type error
    emb2 = tf.math.l2_normalize(np.array([[3, 3, 3]], dtype="float32"), axis=-1)
    emb2 = np.array(emb2)[0]
    search_index2.add(emb2, 3)
    idxs3, embs3 = search_index2.lookup(target, k=3)

    assert len(embs3) == 3
    assert list(idxs3) == [0, 2, 1]


def test_batch_vs_single(tmp_path):
    num_targets = 10
    index_size = 100
    vect_dim = 16

    # gen
    idxs = list(range(index_size))

    targets = tf.math.l2_normalize(np.random.random((num_targets, vect_dim)).astype("float32"), axis=-1)
    targets = np.array(targets)
    embs = tf.math.l2_normalize(np.random.random((index_size, vect_dim)).astype("float32"), axis=-1)
    embs = np.array(embs)

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
        for k in range(len(batch_idxs[i])):
            assert batch_idxs[i][k] == singles_idxs[i][k]


def test_ivfpq():
    # test ivfpq ANN indexing with 100M entries
    num_targets = 10
    index_size = 10000
    vect_dim = 16

    # gen
    idxs = np.array(list(range(index_size)))

    targets = tf.math.l2_normalize(np.random.random((num_targets, vect_dim)).astype("float32"), axis=-1)
    targets = np.array(targets)
    embs = tf.math.l2_normalize(np.random.random((index_size, vect_dim)).astype("float32"), axis=-1)
    embs = np.array(embs)

    search_index = FaissSearch("cosine", vect_dim, algo="ivfpq")
    assert search_index.is_built() == False
    search_index.train_index(embs)
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


def test_reset():
    target = tf.math.l2_normalize(np.array([[1, 1, 2]], dtype="float32"), axis=-1)
    target = np.array(target)[0]
    embs = tf.math.l2_normalize(np.array([[1, 1, 3], [3, 1, 2]], dtype="float32"), axis=-1)
    embs = np.array(embs)

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, out_embs = search_index.lookup(target, k=2)

    print(idxs, out_embs)

    assert len(out_embs) == 2
    assert list(idxs) == [0, 1]
    search_index.reset()
    # only add one
    search_index.add(embs[0], 0)
    idxs, out_embs = search_index.lookup(target, k=2)
    assert len(out_embs) == 1
    assert list(idxs) == [0]
