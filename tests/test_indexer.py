import numpy as np
from tensorflow_similarity.indexer import Indexer


def test_indexer_basic_flow():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.add(embs[0], label=0, data='test')
    indexer.add(embs[1], label=1)

    # lookup
    matches = indexer.single_lookup(target, as_dict=False)

    # check stats
    stats = indexer.stats()
    assert stats['size'] == indexer.size()
    assert indexer.size() == 2

    # check results
    assert len(matches) == 4  # emb, dist, label, data
    assert len(matches[indexer.EMBEDDINGS]) == 2
    assert matches[indexer.LABELS][0] == 0
    assert matches[indexer.DATA][0] == 'test'
    assert list(matches[indexer.EMBEDDINGS][0]) == list(embs[0])
    assert stats['num_lookups'] == 1

def test_indexer_as_dict():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.add(embs[0], label=0, data='test')
    indexer.add(embs[1], label=1)

    # lookup
    matches = indexer.single_lookup(target)

    assert isinstance(matches, list)
    assert matches[0]['distance'] < 0.016

    assert np.array_equal(matches[0]['embedding'], embs[0])
    assert matches[0]['label'] == 0
    assert matches[0]['data'] == 'test'


def test_indexer_batch_add():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.batch_add(embs, [0, 1], data=['test', 'test2'])

    # check results
    matches = indexer.single_lookup(target, as_dict=False)

    assert indexer.size() == 2
    assert len(matches) == 4
    assert len(matches[indexer.EMBEDDINGS]) == 2
    assert matches[indexer.LABELS][0] == 0
    assert matches[indexer.DATA][0] == 'test'
    assert list(matches[indexer.EMBEDDINGS][0]) == list(embs[0])


# def broken_feature_test_indexer_batch_lookup():
#     NUM_ELTS = 100
#     NUM_DIMS = 10
#     K = 3
#     data = np.random.randn(NUM_ELTS, NUM_DIMS).astype(np.float32)
#     indexer = Indexer()
#     indexer.batch_add(data)
#     results = indexer.batch_lookup(data, k=K)
#     indexer.stats()
#     assert len(results) == 100
#     assert len(results[0]) == K


def test_index_reset():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.add(embs[0], label=0)
    indexer.add(embs[1], label=1)
    indexer.add(embs[1], label=2)

    # lookup
    matches = indexer.single_lookup(target, as_dict=False)

    # get stats
    stats = indexer.stats()

    # check results
    assert len(matches) == 4
    assert len(matches[indexer.EMBEDDINGS]) == 3
    assert matches[indexer.LABELS][0] == 0
    assert list(matches[indexer.EMBEDDINGS][0]) == list(embs[0])

    # reset
    indexer.reset()
    stats = indexer.stats()
    assert stats['num_lookups'] == 0
    assert stats['num_items'] == 0

    # do-over
    indexer.add(embs[0], label=42)
    indexer.add(embs[1], label=43)

    matches = indexer.single_lookup(target, as_dict=False)
    stats = indexer.stats()

    assert len(matches) == 4
    assert len(matches[0]) == 2
    assert matches[indexer.LABELS][0] == 42
    assert list(matches[indexer.EMBEDDINGS][0]) == list(embs[0])
    assert stats['num_lookups'] == 1
