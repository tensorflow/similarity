import numpy as np

from tensorflow_similarity.stores import MemoryStore


def build_store(records):
    kv_store = MemoryStore()
    idxs = []
    for r in records:
        idx = kv_store.add(r[0], r[1], r[2])
        idxs.append(idx)
    return kv_store, idxs


def test_in_memory_store_and_retrieve():
    records = [
        [[0.1, 0.2], 1, [0, 0, 0]],
        [[0.2, 0.3], 2, [0, 0, 0]]
    ]

    kv_store, idxs = build_store(records)

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


def test_batch_add():
    embs = np.array([[0.1, 0.2], [0.2, 0.3]])
    lbls = np.array([1, 2])
    data = np.array([[0, 0, 0], [1, 1, 1]])

    kv_store = MemoryStore()
    idxs = kv_store.batch_add(embs, lbls, data)
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert np.array_equal(emb, embs[idx])
        assert np.array_equal(lbl, lbls[idx])
        assert np.array_equal(dt, data[idx])


def test_save_and_reload(tmp_path):
    records = [
        [[0.1, 0.2], 1, [0, 0, 0]],
        [[0.2, 0.3], 2, [0, 0, 0]]
    ]

    kv_store, idxs = build_store(records)
    kv_store.save(str(tmp_path))

    # reload
    reloaded_store = MemoryStore()
    reloaded_store.load(tmp_path)

    assert reloaded_store.size() == 2

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = reloaded_store.get(idx)
        assert np.array_equal(emb, records[idx][0])
        assert np.array_equal(lbl, records[idx][1])
        assert np.array_equal(dt, records[idx][2])
