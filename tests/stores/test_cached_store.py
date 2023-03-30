import os

import numpy as np

from tensorflow_similarity.stores import CachedStore


def build_store(records, path):
    kv_store = CachedStore(path=path)
    idxs = []
    for r in records:
        idx = kv_store.add(r[0], r[1], r[2])
        idxs.append(idx)
    return kv_store, idxs


def test_cached_store_and_retrieve(tmp_path):
    records = [[[0.1, 0.2], 1, [0, 0, 0]], [[0.2, 0.3], 2, [0, 0, 0]]]

    kv_store, idxs = build_store(records, tmp_path)

    # check index numbering
    for gt, idx in enumerate(idxs):
        assert isinstance(idx, int)
        assert gt == idx

    # check reference counting
    assert kv_store.size() == 2

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert emb == records[idx][0]
        assert lbl == records[idx][1]
        assert dt == records[idx][2]


def test_reset(tmp_path):
    records = [[[0.1, 0.2], 1, [0, 0, 0]], [[0.2, 0.3], 2, [0, 0, 0]]]

    kv_store, idxs = build_store(records, tmp_path)

    # check reference counting
    assert kv_store.size() == 2

    kv_store.reset()
    assert kv_store.size() == 0

    kv_store.add(records[0][0], records[0][1], records[0][2])
    assert kv_store.size() == 1


def test_batch_add(tmp_path):
    embs = np.array([[0.1, 0.2], [0.2, 0.3]])
    lbls = np.array([1, 2])
    data = np.array([[0, 0, 0], [1, 1, 1]])

    kv_store = CachedStore(path=tmp_path)
    idxs = kv_store.batch_add(embs, lbls, data)
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert np.array_equal(emb, embs[idx])
        assert np.array_equal(lbl, lbls[idx])
        assert np.array_equal(dt, data[idx])


def test_save_and_reload(tmp_path):
    records = [[[0.1, 0.2], 1, [0, 0, 0]], [[0.2, 0.3], 2, [0, 0, 0]]]

    save_path = tmp_path / "save"
    os.mkdir(save_path)
    obj_path = tmp_path / "obj"
    os.mkdir(obj_path)

    kv_store, idxs = build_store(records, obj_path)
    kv_store.save(save_path)

    # reload
    reloaded_store = CachedStore()
    print(f"loading from {save_path}")
    reloaded_store.load(save_path)

    assert reloaded_store.size() == 2

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = reloaded_store.get(idx)
        assert np.array_equal(emb, records[idx][0])
        assert np.array_equal(lbl, records[idx][1])
        assert np.array_equal(dt, records[idx][2])
