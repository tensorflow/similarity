import numpy as np
from tensorflow_similarity.indexer import Indexer


def test_indexer_basic_flow():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.add(embs[0], label=0, data='test')
    indexer.add(embs[1], label=1)
    indexer.build()

    # check results
    matches = indexer.lookup(target)
    assert len(matches) == 2
    assert matches[0]['label'] == 0
    assert matches[0]['data'] == 'test'
    assert list(matches[0]['embedding']) == list(embs[0])


def test_indexer_batch_add():

    target = np.array([1, 1, 2], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer()

    # index data
    indexer.batch_add(embs, [0, 1], data=['test', 'test2'])
    indexer.build()

    # check results
    matches = indexer.lookup(target)
    assert len(matches) == 2
    assert matches[0]['label'] == 0
    assert matches[0]['data'] == 'test'
    assert list(matches[0]['embedding']) == list(embs[0])
