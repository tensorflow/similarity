import numpy as np
import tensorflow as tf

from tensorflow_similarity.search.faiss import FaissSearch


def test_index_match():
    target = tf.constant([[1, 1, 2]], dtype="float32")
    target = tf.math.l2_normalize(target, axis=-1)[0]
    embs = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="float32")
    embs = tf.math.l2_normalize(embs, axis=-1)

    search_index = FaissSearch("cosine", 3, algo="flat")
    search_index.add(embs[0], 0)
    search_index.add(embs[1], 1)

    idxs, embs = search_index.lookup(target, k=2)

    assert len(embs) == 2
    assert list(idxs) == [0, 1]


def test_index_save(tmp_path):
    target = tf.constant([[1, 1, 2]], dtype="float32")
    target = tf.math.l2_normalize(target, axis=-1)[0]
    embs = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="float32")
    embs = tf.math.l2_normalize(embs, axis=-1)
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
    emb2 = tf.constant([[3, 3, 3]], dtype="float32")
    emb2 = tf.math.l2_normalize(emb2, axis=-1)[0]
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

    targets = tf.random.uniform((num_targets, vect_dim), dtype="float32")
    targets = tf.math.l2_normalize(targets, axis=-1)
    embs = tf.random.uniform((index_size, vect_dim), dtype="float32")
    embs = tf.math.l2_normalize(embs, axis=-1)

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

    targets = tf.random.uniform((num_targets, vect_dim), dtype="float32")
    targets = tf.math.l2_normalize(targets, axis=-1)
    embs = tf.random.uniform((index_size, vect_dim), dtype="float32")
    embs = tf.math.l2_normalize(embs, axis=-1)

    search_index = FaissSearch("cosine", vect_dim, algo="ivfpq")
    assert not search_index.is_built()
    search_index.train_index(embs)
    assert search_index.is_built()
    last_idx = 0
    for i in range(1000):
        idxs = tf.range(last_idx, last_idx + index_size)
        embs = tf.random.uniform((index_size, vect_dim), dtype="float32")
        embs = tf.math.l2_normalize(embs, axis=-1)
        last_idx += index_size
        search_index.batch_add(embs, idxs)
    found_idxs, found_dists = search_index.batch_lookup(targets, 2)
    assert len(found_idxs) == 10
    assert len(found_idxs[0]) == 2


def test_reset():
    target = tf.constant([[1, 1, 2]], dtype="float32")
    target = tf.math.l2_normalize(target, axis=-1)[0]
    embs = tf.constant([[1, 1, 3], [3, 1, 2]], dtype="float32")
    embs = tf.math.l2_normalize(embs, axis=-1)

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
