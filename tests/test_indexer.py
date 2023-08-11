import numpy as np
import tensorflow as tf

from tensorflow_similarity.indexer import Indexer
from tensorflow_similarity.search import FaissSearch, LinearSearch
from tensorflow_similarity.stores import CachedStore

from . import DATA_DIR


class IndexerTest(tf.test.TestCase):
    def test_calibration(self):
        # CALIB TEST
        SIZE = 20

        FNAME = str(DATA_DIR / "mnist_fashion_embeddings.npz")
        data = np.load(FNAME, allow_pickle=True)
        thresholds_targets = {"0.5": 0.5}

        index = Indexer(3)
        index.batch_add(data["embeddings_idx"][:SIZE], labels=data["y_idx"][:SIZE])
        calibration = index.calibrate(
            data["embeddings_cal"][:SIZE],
            data["y_cal"][:SIZE],
            thresholds_targets,
            verbose=1,
        )
        # assert 'vl' in cutpoints
        self.assertIn("optimal", calibration.cutpoints)
        self.assertIn("0.5", calibration.cutpoints)
        self.assertEqual(
            len(calibration.thresholds["distance"]),
            len(calibration.thresholds["value"]),
        )
        self.assertTrue(index.is_calibrated)

    def test_indexer_basic_flow(self):
        prediction = np.array([[1, 1, 2]], dtype="float32")
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)

        # index data
        indexer.batch_add(embs, labels=[0, 1], data=["test", "test2"])

        # lookup
        matches = indexer.single_lookup(prediction)

        self.assertIsInstance(matches, list)
        self.assertLess(matches[0].distance, 0.016)
        self.assertIsNotNone(matches[0].embedding)

        self.assertIsNone(np.testing.assert_array_equal(matches[0].embedding, embs[0]))
        self.assertEqual(matches[0].label, 0)
        self.assertEqual(matches[0].data, "test")

    def test_indexer_batch_add(self):
        prediction = np.array([[1, 1, 2]], dtype="float32")
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)

        # index data
        indexer.batch_add(embs, [0, 1], data=["test", "test2"])
        self.assertEqual(indexer.size(), 2)
        # check results
        matches = indexer.single_lookup(prediction)

        self.assertIsInstance(matches, list)
        self.assertLess(matches[0].distance, 0.016)

        self.assertIsNone(np.testing.assert_array_equal(matches[0].embedding, embs[0]))
        self.assertEqual(matches[0].label, 0)
        self.assertEqual(matches[0].data, "test")

    def test_multiple_add(self):
        # arrays of preds which contains a single embedding list(list(embedding))
        predictions = np.array([[[1, 1, 3]], [[3, 1, 2]]], dtype="float32")

        indexer = Indexer(3)
        indexer.add(predictions[0])
        self.assertEqual(indexer.size(), 1)

        indexer.add(predictions[1])
        self.assertEqual(indexer.size(), 2)

    def test_multiple_add_mix_data(self):
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)
        indexer.batch_add(embs)
        self.assertEqual(indexer.size(), 2)

        indexer.batch_add(embs, data=embs)
        self.assertEqual(indexer.size(), 4)

    def test_reload(self):
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 2)

        path = self.get_temp_dir()

        # save
        indexer.save(path)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 2)

        # add more
        indexer2.batch_add(embs, data=embs)
        self.assertEqual(indexer2.size(), 4)

    def test_uncompress_reload(self):
        "Ensure uncompressed index work"

        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 2)

        # save
        path = self.get_temp_dir()
        indexer.save(path, compression=False)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 2)

    def test_linear_search_reload(self):
        "Ensure the save and load of custom search and store work"
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")
        search = LinearSearch("cosine", 3)
        store = CachedStore()

        indexer = Indexer(3, search=search, kv_store=store)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 2)

        # save
        path = self.get_temp_dir()
        indexer.save(path, compression=False)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 2)

    def test_faiss_search_reload(self):
        "Ensure the save and load of Faiss search and store work"
        embs = np.random.random((1024, 8)).astype(np.float32)
        search = FaissSearch("cosine", 8, m=4, nlist=2)
        store = CachedStore()

        indexer = Indexer(8, search=search, kv_store=store)
        indexer.build_index(embs)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 1024)

        # save
        path = self.get_temp_dir()
        indexer.save(path, compression=False)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 1024)

    def test_index_reset(self):
        prediction = np.array([[1, 1, 2]], dtype="float32")
        embs = np.array([[1, 1, 3], [3, 1, 2], [3, 2, 3]], dtype="float32")

        indexer = Indexer(3)

        # index data
        indexer.batch_add(embs, labels=[0, 1, 2])

        # lookup
        matches = indexer.single_lookup(prediction)

        # get stats
        stats = indexer.stats()

        # check results
        self.assertLen(matches, 3)
        self.assertLen(matches[0].embedding, 3)
        self.assertEqual(matches[0].label, 0)
        self.assertEqual(list(matches[0].embedding), list(embs[0]))

        # reset
        indexer.reset()
        stats = indexer.stats()
        self.assertEqual(stats["num_lookups"], 0)
        self.assertEqual(stats["num_items"], 0)

        # do-over
        indexer.add([embs[0]], label=42)
        indexer.add([embs[1]], label=43)

        matches = indexer.single_lookup(prediction)
        stats = indexer.stats()

        self.assertLen(matches, 2)
        self.assertLen(matches[0].embedding, 3)
        self.assertEqual(matches[0].label, 42)
        self.assertEqual(matches[1].label, 43)
        self.assertAllEqual(matches[0].embedding, embs[0])
        self.assertAllEqual(matches[1].embedding, embs[1])
        self.assertEqual(stats["num_lookups"], 1)

    def test_indexer_batch_ops(self):
        NUM_ELTS = 100
        NUM_DIMS = 10
        K = 3
        data = np.random.randn(NUM_ELTS, NUM_DIMS).astype(np.float32)
        indexer = Indexer(NUM_DIMS)
        indexer.batch_add(data)
        results = indexer.batch_lookup(data, k=K)
        indexer.stats()
        self.assertLen(results, 100)
        self.assertLen(results[0], K)

    def test_single_vs_batch_ops(self):
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
            self.assertEqual(single_results[idx][0].label, batch_results[idx][0].label)

    def test_multiple_add_mix_data(self):
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)
        indexer.batch_add(embs)
        self.assertEqual(indexer.size(), 2)

        indexer.batch_add(embs, data=embs)
        self.assertEqual(indexer.size(), 4)

    def test_reload(self):
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 2)

        # save
        path = self.get_temp_dir()
        indexer.save(path)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 2)

        # add more
        indexer2.batch_add(embs, data=embs)
        self.assertEqual(indexer2.size(), 4)

    def test_uncompress_reload(self):
        "Ensure uncompressed index work"

        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")

        indexer = Indexer(3)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 2)

        # save
        path = self.get_temp_dir()
        indexer.save(path, compression=False)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 2)

    def test_linear_search_reload(self):
        "Ensure the save and load of custom search and store work"
        embs = np.array([[1, 1, 3], [3, 1, 2]], dtype="float32")
        search = LinearSearch("cosine", 3)
        store = CachedStore()

        indexer = Indexer(3, search=search, kv_store=store)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 2)

        # save
        path = self.get_temp_dir()
        indexer.save(path, compression=False)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 2)

    def test_faiss_search_reload(self):
        "Ensure the save and load of Faiss search and store work"
        embs = np.random.random((1024, 8)).astype(np.float32)
        search = FaissSearch("cosine", 8, m=4, nlist=2)
        store = CachedStore()

        indexer = Indexer(8, search=search, kv_store=store)
        indexer.build_index(embs)
        indexer.batch_add(embs, verbose=0)
        self.assertEqual(indexer.size(), 1024)

        # save
        path = self.get_temp_dir()
        indexer.save(path, compression=False)

        # reload
        indexer2 = Indexer.load(path)
        self.assertEqual(indexer2.size(), 1024)

    def test_index_reset(self):
        prediction = np.array([[1, 1, 2]], dtype="float32")
        embs = np.array([[1, 1, 3], [3, 1, 2], [3, 2, 3]], dtype="float32")

        indexer = Indexer(3)

        # index data
        indexer.batch_add(embs, labels=[0, 1, 2])

        # lookup
        matches = indexer.single_lookup(prediction)

        # get stats
        stats = indexer.stats()

        # check results
        self.assertLen(matches, 3)
        self.assertLen(matches[0].embedding, 3)
        self.assertEqual(matches[0].label, 0)
        self.assertEqual(list(matches[0].embedding), list(embs[0]))

        # reset
        indexer.reset()
        stats = indexer.stats()
        self.assertEqual(stats["num_lookups"], 0)
        self.assertEqual(stats["num_items"], 0)

        # do-over
        indexer.add([embs[0]], label=42)
        indexer.add([embs[1]], label=43)

        matches = indexer.single_lookup(prediction)
        stats = indexer.stats()

        self.assertLen(matches, 2)
        self.assertLen(matches[0].embedding, 3)
        self.assertEqual(matches[0].label, 42)
        self.assertEqual(matches[1].label, 43)
        self.assertEqual(list(matches[0].embedding), list(embs[0]))
        self.assertEqual(list(matches[1].embedding), list(embs[1]))
        self.assertEqual(stats["num_lookups"], 1)

    def test_indexer_batch_ops(self):
        NUM_ELTS = 100
        NUM_DIMS = 10
        K = 3
        data = np.random.randn(NUM_ELTS, NUM_DIMS).astype(np.float32)
        indexer = Indexer(NUM_DIMS)
        indexer.batch_add(data)
        results = indexer.batch_lookup(data, k=K)
        indexer.stats()
        self.assertLen(results, 100)
        self.assertLen(results[0], K)

    def test_single_vs_batch_ops(self):
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
            self.assertEqual(single_results[idx][0].label, batch_results[idx][0].label)


if __name__ == "__main__":
    tf.test.main()
