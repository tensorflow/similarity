import numpy as np
from tensorflow_similarity.tables import MemoryTable


def build_table(records):
    index_table = MemoryTable()
    idxs = []
    for r in records:
        idx = index_table.add(r[0], r[1], r[2])
        idxs.append(idx)
    return index_table, idxs


def test_in_memory_store_and_retrieve():
    records = [
        [[0.1, 0.2], 1, [0, 0, 0]],
        [[0.2, 0.3], 2, [0, 0, 0]]
    ]

    index_table, idxs = build_table(records)

    # check index numbering
    for gt, idx in enumerate(idxs):
        assert isinstance(idx, int)
        assert gt == idx

    # check reference counting
    assert index_table.size() == 2

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = index_table.get(idx)
        assert emb == records[idx][0]
        assert lbl == records[idx][1]
        assert dt == records[idx][2]


def test_batch_add():
    embs = np.array([[0.1, 0.2], [0.2, 0.3]])
    lbls = np.array([1, 2])
    data = np.array([[0, 0, 0], [1, 1, 1]])

    index_table = MemoryTable()
    idxs = index_table.batch_add(embs, lbls, data)
    for idx in idxs:
        emb, lbl, dt = index_table.get(idx)
        assert np.array_equal(emb, embs[idx])
        assert np.array_equal(lbl, lbls[idx])
        assert np.array_equal(dt, data[idx])


def test_save_and_reload(tmp_path):
    records = [
        [[0.1, 0.2], 1, [0, 0, 0]],
        [[0.2, 0.3], 2, [0, 0, 0]]
    ]

    index_table, idxs = build_table(records)
    index_table.save(str(tmp_path))

    # reload
    reloaded_table = MemoryTable()
    reloaded_table.load(tmp_path)

    assert reloaded_table.size() == 2

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = reloaded_table.get(idx)
        assert np.array_equal(emb, records[idx][0])
        assert np.array_equal(lbl, records[idx][1])
        assert np.array_equal(dt, records[idx][2])
