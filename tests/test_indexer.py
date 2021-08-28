import numpy as np
from tensorflow_similarity.indexer import Indexer
from . import DATA_DIR


def test_calibration():
    # CALIB TEST
    SIZE = 20

    FNAME = str(DATA_DIR / 'mnist_fashion_embeddings.npz')
    data = np.load(FNAME, allow_pickle=True)
    thresholds_targets = {'0.5': 0.5}

    index = Indexer(3)
    index.batch_add(data['embeddings_idx'][:SIZE], labels=data['y_idx'][:SIZE])
    calibration = index.calibrate(data['embeddings_cal'][:SIZE],
                                  data['y_cal'][:SIZE],
                                  thresholds_targets,
                                  verbose=1)
    # assert 'vl' in cutpoints
    assert 'optimal' in calibration.cutpoints
    assert '0.5' in calibration.cutpoints
    assert len(calibration.thresholds['distance']) == len(
        calibration.thresholds['value'])
    assert index.is_calibrated


def test_indexer_basic_flow():

    prediction = np.array([[1, 1, 2]], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer(3)

    # index data
    indexer.batch_add(embs, labels=[0, 1], data=['test', 'test2'])

    # lookup
    matches = indexer.single_lookup(prediction)

    assert isinstance(matches, list)
    assert matches[0].distance < 0.016

    assert np.array_equal(matches[0].embedding, embs[0])
    assert matches[0].label == 0
    assert matches[0].data == 'test'


def test_indexer_batch_add():

    prediction = np.array([[1, 1, 2]], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer(3)

    # index data
    indexer.batch_add(embs, [0, 1], data=['test', 'test2'])
    assert indexer.size() == 2
    # check results
    matches = indexer.single_lookup(prediction)

    assert isinstance(matches, list)
    assert matches[0].distance < 0.016

    assert np.array_equal(matches[0].embedding, embs[0])
    assert matches[0].label == 0
    assert matches[0].data == 'test'


def test_multiple_add():

    # arrays of preds which contains a single embedding list(list(embedding))
    predictions = np.array([[[1, 1, 3]], [[3, 1, 2]]], dtype='float32')

    indexer = Indexer(3)
    indexer.add(predictions[0])
    assert indexer.size() == 1

    indexer.add(predictions[1])
    assert indexer.size() == 2


def test_multiple_add_mix_data():

    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer(3)
    indexer.batch_add(embs)
    assert indexer.size() == 2

    indexer.batch_add(embs, data=embs)
    assert indexer.size() == 4


def test_reload(tmp_path):

    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer(3)
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


def test_uncompress_reload(tmp_path):
    "Ensure uncompressed index work"

    embs = np.array([[1, 1, 3], [3, 1, 2]], dtype='float32')

    indexer = Indexer(3)
    indexer.batch_add(embs, verbose=0)
    assert indexer.size() == 2

    # save
    path = tmp_path / "test_save_and_add/"
    indexer.save(path, compression=False)

    # reload
    indexer2 = Indexer.load(path)
    assert indexer2.size() == 2


def test_index_reset():

    prediction = np.array([[1, 1, 2]], dtype='float32')
    embs = np.array([[1, 1, 3], [3, 1, 2], [3, 2, 3]], dtype='float32')

    indexer = Indexer(3)

    # index data
    indexer.batch_add(embs, labels=[0, 1, 2])

    # lookup
    matches = indexer.single_lookup(prediction)

    # get stats
    stats = indexer.stats()

    # check results
    assert len(matches) == 3
    assert len(matches[0].embedding) == 3
    assert matches[0].label == 0
    assert list(matches[0].embedding) == list(embs[0])

    # reset
    indexer.reset()
    stats = indexer.stats()
    assert stats['num_lookups'] == 0
    assert stats['num_items'] == 0

    # do-over
    indexer.add([embs[0]], label=42)
    indexer.add([embs[1]], label=43)

    matches = indexer.single_lookup(prediction)
    stats = indexer.stats()

    assert len(matches) == 2
    assert len(matches[0].embedding) == 3
    assert matches[0].label == 42
    assert matches[1].label == 43
    assert list(matches[0].embedding) == list(embs[0])
    assert list(matches[1].embedding) == list(embs[1])
    assert stats['num_lookups'] == 1


def test_indexer_batch_ops():
    NUM_ELTS = 100
    NUM_DIMS = 10
    K = 3
    data = np.random.randn(NUM_ELTS, NUM_DIMS).astype(np.float32)
    indexer = Indexer(NUM_DIMS)
    indexer.batch_add(data)
    results = indexer.batch_lookup(data, k=K)
    indexer.stats()
    assert len(results) == 100
    assert len(results[0]) == K


def test_single_vs_batch_ops():
    "ensure batch and single ops are consistent"
    NUM_ELTS = 100
    NUM_DIMS = 10
    K = 3
    data = np.random.randn(NUM_ELTS, NUM_DIMS).astype(np.float32)
    indexer = Indexer(NUM_DIMS)
    indexer.batch_add(data)
    batch_results = indexer.batch_lookup(data, k=K)

    single_results = []
    for d in data:
        single_results.append(indexer.single_lookup([d], k=K))

    for idx in range(len(single_results)):
        assert single_results[idx][0].label == batch_results[idx][0].label
