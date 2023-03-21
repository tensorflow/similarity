import pickle
from unittest.mock import MagicMock, patch

import numpy as np

from tensorflow_similarity.stores import RedisStore


def build_store(records):
    kv_store = RedisStore()
    idxs = []
    for r in records:
        idx = kv_store.add(r[0], r[1], r[2])
        idxs.append(idx)
    return kv_store, idxs


@patch("redis.Redis", return_value=MagicMock())
def test_store_and_retrieve(mock_redis):
    records = [[[0.1, 0.2], 1, [0, 0, 0]], [[0.2, 0.3], 2, [0, 0, 0]]]
    serialized_records = [pickle.dumps(x) for x in records]
    mock_redis.return_value.get.side_effect = serialized_records
    mock_redis.return_value.incr.side_effect = [1, 2, 3, 4, 5]

    kv_store, idxs = build_store(records)

    # check index numbering
    for gt, idx in enumerate(idxs):
        assert isinstance(idx, int)
        assert gt == idx

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert emb == records[idx][0]
        assert lbl == records[idx][1]
        assert dt == records[idx][2]


@patch("redis.Redis", return_value=MagicMock())
def test_batch_add(mock_redis):
    embs = np.array([[0.1, 0.2], [0.2, 0.3]])
    lbls = np.array([1, 2])
    data = np.array([[0, 0, 0], [1, 1, 1]])

    records = [[embs[i], lbls[i], data[i]] for i in range(2)]

    serialized_records = [pickle.dumps(r) for r in records]
    mock_redis.return_value.get.side_effect = serialized_records
    mock_redis.return_value.incr.side_effect = [1, 2, 3, 4, 5]

    kv_store = RedisStore()
    idxs = kv_store.batch_add(embs, lbls, data)
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert np.array_equal(emb, embs[idx])
        assert np.array_equal(lbl, lbls[idx])
        assert np.array_equal(dt, data[idx])
