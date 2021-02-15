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
    matches = indexer.single_lookup(target)

    assert isinstance(matches, list)
    assert matches[0]['distance'] < 0.016

    assert np.array_equal(matches[0]['embedding'], embs[0])
    assert matches[0]['label'] == 0
    assert matches[0]['data'] == 'test'


def test_multiple_add():

    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()
    indexer.batch_add(embs)
    assert indexer.size() == 2

    indexer.batch_add(embs)
    assert indexer.size() == 4

    indexer.add(embs[0])


def test_multiple_add_mix_data():

    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()
    indexer.batch_add(embs)
    assert indexer.size() == 2

    indexer.batch_add(embs, data=embs)
    assert indexer.size() == 4


def test_reload(tmp_path):
    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()
    indexer.batch_add(embs, verbose=0)
    assert indexer.size() == 2

    # save
    path = tmp_path / "test_save_and_add/"
    indexer.save(path)

    # reload
    indexer2 = Indexer.load(path)
    assert indexer2.size() == 2

    # add more
    indexer2.batch_add(embs, data=embs)
    assert indexer2.size() == 4


def test_index_reset():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.add(embs[0], label=0)
    indexer.add(embs[1], label=1)
    indexer.add(embs[1], label=2)

    # lookup
    matches = indexer.single_lookup(target)

    # get stats
    stats = indexer.stats()

    # check results
    assert len(matches) == 3
    assert len(matches[0]['embedding']) == 3
    assert matches[0]['label'] == 0
    assert list(matches[0]['embedding']) == list(embs[0])

    # reset
    indexer.reset()
    stats = indexer.stats()
    assert stats['num_lookups'] == 0
    assert stats['num_items'] == 0

    # do-over
    indexer.add(embs[0], label=42)
    indexer.add(embs[1], label=43)

    matches = indexer.single_lookup(target)
    stats = indexer.stats()

    assert len(matches) == 2
    assert len(matches[0]['embedding']) == 3
    assert matches[0]['label'] == 42
    assert matches[1]['label'] == 43
    assert list(matches[0]['embedding']) == list(embs[0])
    assert list(matches[1]['embedding']) == list(embs[1])
    assert stats['num_lookups'] == 1

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