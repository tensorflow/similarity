import tensorflow as tf

from tensorflow_similarity.samplers import TFDatasetMultiShotMemorySampler


class DatasetSamplersTest(tf.test.TestCase):
    def test_basic(self):
        dataset_name = "mnist"
        sampler = TFDatasetMultiShotMemorySampler(dataset_name=dataset_name, classes_per_batch=10)
        batch = sampler.generate_batch(42)
        self.assertEqual(batch[0].shape, (20, 28, 28, 1))

    def test_wrong_key(self):
        dataset_name = "mnist"

        # X
        with self.assertRaises(ValueError):
            TFDatasetMultiShotMemorySampler(dataset_name=dataset_name, classes_per_batch=4, x_key="error")
        # Y
        with self.assertRaises(ValueError):
            TFDatasetMultiShotMemorySampler(dataset_name=dataset_name, classes_per_batch=4, y_key="error")
