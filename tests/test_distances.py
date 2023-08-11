import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.distances import (
    DISTANCES,
    CosineDistance,
    EuclideanDistance,
    InnerProductSimilarity,
    ManhattanDistance,
    SNRDistance,
    distance_canonicalizer,
)


class DistancesTest(tf.test.TestCase):
    def test_distance_mapping(self):
        for d in DISTANCES:
            # self naming
            d2 = distance_canonicalizer(d.name)
            self.assertEqual(d2.name, d.name)

            # aliases
            for a in d.aliases:
                d2 = distance_canonicalizer(a)
                self.assertEqual(d2.name, d.name)

    def test_distance_passthrough(self):
        # Canonilizer is expected to return distance object as is
        d = EuclideanDistance()
        d2 = distance_canonicalizer(d)
        self.assertEqual(d, d2)

    def test_non_existing_distance(self):
        with pytest.raises(ValueError):
            distance_canonicalizer("notadistance")

    def test_inner_product_similarity(self):
        # pairwise
        a = tf.convert_to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        d = InnerProductSimilarity()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 12.0)

    def test_inner_product_key(self):
        a = tf.convert_to_tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        b = tf.convert_to_tensor([[0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        d = InnerProductSimilarity()
        vals = d(a, b)
        expected = tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 3.0], [0.0, 3.0, 9.0]])

        self.assertAllEqual(vals, expected)

    def test_inner_product_opposite(self):
        a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
        d = InnerProductSimilarity()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 2.0)

    def test_inner_product_vals(self):
        a = tf.nn.l2_normalize([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]], axis=-1)
        d = InnerProductSimilarity()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))
        self.assertEqual(vals[0][0], 1)
        self.assertEqual(vals[0][1], 0.68138516)

    def test_cosine_key(self):
        a = tf.convert_to_tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        b = tf.convert_to_tensor([[0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        d = CosineDistance()
        vals = d(a, b)
        expected = tf.constant([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertAllClose(vals, expected)

    def test_cosine_same(self):
        # pairwise
        a = tf.convert_to_tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        d = CosineDistance()
        vals = d(a, a)
        val = tf.shape(vals)
        self.assertAllEqual(val, (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 0)

    def test_cosine_opposite(self):
        a = tf.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]])
        d = CosineDistance()
        vals = d(a, a)
        val = tf.shape(vals)
        self.assertAllEqual(val, (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 2.0)

    def test_cosine_vals(self):
        a = tf.nn.l2_normalize([[0.1, 0.3, 0.2], [0.0, 0.1, 0.5]], axis=-1)
        d = CosineDistance()
        vals = d(a, a)
        self.assertEqual(vals[0][0], 0)
        self.assertEqual(vals[0][1], 0.31861484)

    def test_euclidean(self):
        a = tf.convert_to_tensor([[0.0, 3.0], [4.0, 0.0]])
        d = EuclideanDistance()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 10.0)

    def test_euclidean_key(self):
        a = tf.convert_to_tensor(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        b = tf.convert_to_tensor(
            [
                [2.0, 1.0],
                [1.0, 1.0],
            ]
        )
        d = EuclideanDistance()
        vals = d(a, b)
        expected = tf.constant([[2.0, 1.0], [2.0, 1.0], [1.0, 0.0]])
        self.assertAllEqual(vals, expected)

    def test_euclidean_same(self):
        a = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        d = EuclideanDistance()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 0.0)

    def test_euclidean_opposite(self):
        a = tf.convert_to_tensor([[0.0, 1.0], [0.0, -1.0]])
        d = EuclideanDistance()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))
        self.assertAlmostEqual(tf.reduce_sum(vals), 4.0)

    def test_manhattan(self):
        a = tf.convert_to_tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        d = ManhattanDistance()
        vals = d(a, a)
        expected = tf.constant(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 4.0],
                [2.0, 1.0, 0.0, 3.0],
                [3.0, 4.0, 3.0, 0.0],
            ]
        )
        self.assertAllEqual(tf.shape(vals), (4, 4))
        self.assertAllEqual(vals, expected)

    def test_manhattan_key(self):
        a = tf.convert_to_tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        b = tf.convert_to_tensor([[0.0, 1.0], [1.0, 1.0], [3.0, 0.0]])
        d = ManhattanDistance()
        vals = d(a, b)
        expected = tf.constant([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [1.0, 0.0, 3.0], [4.0, 3.0, 0.0]])
        self.assertAllEqual(vals, expected)

    def test_manhattan_same(self):
        a = tf.convert_to_tensor([[1.0, 1.0], [1.0, 1.0]])
        d = ManhattanDistance()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))

        self.assertAlmostEqual(tf.reduce_sum(vals), 0.0)

    def test_manhattan_opposite(self):
        a = tf.convert_to_tensor([[0.0, 1.0], [0.0, -1.0]])
        d = ManhattanDistance()
        vals = d(a, a)
        self.assertAllEqual(tf.shape(vals), (2, 2))

        self.assertAlmostEqual(tf.reduce_sum(vals), 4.0)

    def test_snr_dist(self):
        """
        Comparing SNRDistance with simple loop based implementation
        of SNR distance.
        """
        num_inputs = 3
        dims = 5
        x = np.random.uniform(0, 1, (num_inputs, dims))

        # Computing SNR distance values using loop
        snr_pairs = []
        for i in range(num_inputs):
            row = []
            for j in range(num_inputs):
                dist = np.var(x[i] - x[j]) / np.var(x[i])
                row.append(dist)
            snr_pairs.append(row)
        snr_pairs = np.array(snr_pairs)

        x = tf.convert_to_tensor(x)
        snr_distances = SNRDistance()(x, x).numpy()
        self.assertAllGreaterEqual(snr_distances, 0)
        diff = snr_distances - snr_pairs
        self.assertAllLess(tf.math.abs(diff), 1e-4)

    def test_snr_dist_key(self):
        """Comparing SNRDistance with simple loop based implementation

        of SNR distance for 2 different embedding tensors.
        """

        num_inputs = 3
        num_inputs2 = 2
        dims = 5
        x = np.random.uniform(0, 1, (num_inputs, dims))
        x2 = np.random.uniform(0, 1, (num_inputs2, dims))

        # Computing SNR distance values using loop
        snr_pairs = []
        for i in range(num_inputs):
            row = []
            for j in range(num_inputs2):
                dist = np.var(x[i] - x2[j]) / np.var(x[i])
                row.append(dist)
            snr_pairs.append(row)
        snr_pairs = np.array(snr_pairs)

        x = tf.convert_to_tensor(x)
        snr_distances = SNRDistance()(x, x2).numpy()
        self.assertAllGreaterEqual(snr_distances, 0)
        diff = snr_distances - snr_pairs
        self.assertAllLess(tf.math.abs(diff), 1e-4)


if __name__ == "__main__":
    tf.test.main()
