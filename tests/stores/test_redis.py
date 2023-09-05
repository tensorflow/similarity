import numpy as np
from mock import patch

from tensorflow_similarity.stores.redis import RedisStore


class MockRedis:
    def __init__(self, *args, **kwargs):
        self.flushdb()

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None  # return nil

    def set(self, key, value, *args, **kwargs):
        self.cache[key] = value
        return "OK"

    def flushdb(self):
        self.cache = {}
        return "OK"

    def incr(self, key):
        if key in self.cache:
            self.cache[key] += 1
        else:
            self.cache[key] = 1
        return self.cache[key]


@patch("redis.Redis", MockRedis)
def test_store_and_retrieve():
    recs = [[[0.1, 0.2], 1, [0, 0, 0]], [[0.2, 0.3], 2, [0, 0, 0]]]

    kv_store = RedisStore()
    idxs = []
    for r in recs:
        idx = kv_store.add(r[0], r[1], r[2])
        idxs.append(idx)

    # check index numbering
    for gt, idx in enumerate(idxs):
        assert isinstance(idx, int)
        assert gt == idx

    # get back three elements
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert emb == recs[idx][0]
        assert lbl == recs[idx][1]
        assert dt == recs[idx][2]


@patch("redis.Redis", MockRedis)
def test_batch_add():
    embs = np.array([[0.1, 0.2], [0.2, 0.3]])
    lbls = np.array([1, 2])
    data = np.array([[0, 0, 0], [1, 1, 1]])

    kv_store = RedisStore()
    idxs = kv_store.batch_add(embs, lbls, data)
    for idx in idxs:
        emb, lbl, dt = kv_store.get(idx)
        assert np.array_equal(emb, embs[idx])
        assert np.array_equal(lbl, lbls[idx])
        assert np.array_equal(dt, data[idx])
