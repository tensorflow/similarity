import tensorflow as tf
from absl.testing import parameterized

from tensorflow_similarity.algebra import build_masks, masked_max, masked_min


class LossTest(tf.test.TestCase, parameterized.TestCase):
    def test_mask(self):
        batch_size = 16
        labels = tf.random.uniform((batch_size, 1), 0, 10, dtype=tf.int32)
        positive_mask, negative_mask = build_masks(labels, labels, batch_size)
        self.assertFalse(positive_mask[0][0])
        self.assertFalse(positive_mask[5][5])

        combined = tf.math.logical_or(negative_mask, positive_mask)
        self.assertFalse(combined[0][0])
        for i in range(1, batch_size):
            self.assertTrue(combined[0][i])
            self.assertTrue(combined[i][0])

    @parameterized.named_parameters(
        {
            "testcase_name": "_rank1_labels",
            "query_labels": [1, 2, 0],
            "key_labels": [1, 2, 3, 0],
            "batch_size": 3,
        },
        {
            "testcase_name": "_rank2_labels",
            "query_labels": [[1], [2], [0]],
            "key_labels": [[1], [2], [3], [0]],
            "batch_size": 3,
        },
    )
    def test_mask_non_square(self, query_labels, key_labels, batch_size):
        query_labels = tf.constant(
            query_labels,
            dtype=tf.int32,
        )

        key_labels = tf.constant(
            key_labels,
            dtype=tf.int32,
        )

        positive_mask_nodiag, negative_mask_nodiag = build_masks(query_labels, key_labels, batch_size)
        positive_mask_wdiag, negative_mask_wdiag = build_masks(
            query_labels,
            key_labels,
            batch_size,
            remove_diagonal=False,
        )

        target_positive_mask_nodiag = tf.constant(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, True],
            ]
        )
        self.assertAllEqual(positive_mask_nodiag, target_positive_mask_nodiag)

        target_positive_mask_wdiag = tf.constant(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, True],
            ]
        )

        self.assertAllEqual(positive_mask_wdiag, target_positive_mask_wdiag)
        self.assertAllEqual(negative_mask_wdiag, negative_mask_nodiag)

    def test_masked_max(self):
        distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 2.0, 1.0, 0.0]], dtype=tf.float32)
        mask = tf.constant([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=tf.float32)
        vals, arg_max = masked_max(distances, mask)

        self.assertEqual(vals.shape, (2, 1))
        self.assertEqual(arg_max.shape, (2,))
        self.assertEqual(vals[0], [3.0])
        self.assertEqual(vals[1], [2.0])
        self.assertEqual(arg_max[0], [2])
        self.assertEqual(arg_max[1], [1])

    def test_arg_max_all_unmasked_vals_lt_zero(self):
        # Ensure reduce_max works when all unmasked vals < 0.0.
        distances = tf.constant(
            [[-7.0, -2.0, 7.0, -9.0], [-7.0, 1e-05, 7.0, -9.0]],
            dtype=tf.float32,
        )
        mask = tf.constant([[0, 0, 0, 1], [0, 1, 0, 0]], dtype=tf.float32)
        vals, arg_max = masked_max(distances, mask)

        self.assertEqual(vals.shape, (2, 1))
        self.assertEqual(arg_max.shape, (2,))
        self.assertEqual(vals[0], [-9.0])
        self.assertEqual(vals[1], [1e-05])
        self.assertEqual(arg_max[0], [3])
        self.assertEqual(arg_max[1], [1])

    def test_masked_min(self):
        distances = tf.constant([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
        mask = tf.constant([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=tf.float32)
        vals, arg_min = masked_min(distances, mask)

        self.assertEqual(vals.shape, (2, 1))
        self.assertEqual(arg_min.shape, (2,))
        self.assertEqual(vals[0], [2.0])
        self.assertEqual(vals[1], [1.0])
        self.assertEqual(arg_min[0], [1])
        self.assertEqual(arg_min[1], [2])

    def test_arg_min_all_unmasked_vals_gt_zero(self):
        # Ensure reduce_max works when all unmasked vals > 0.0.
        distances = tf.constant(
            [[-7.0, -2.0, 7.0, -9.0], [-1e-06, -2.0, 7.0, -9.0]],
            dtype=tf.float32,
        )
        mask = tf.constant([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=tf.float32)
        vals, arg_min = masked_min(distances, mask)

        self.assertEqual(vals.shape, (2, 1))
        self.assertEqual(arg_min.shape, (2,))
        self.assertEqual(vals[0], [7.0])
        self.assertEqual(vals[1], [-1e-06])
        self.assertEqual(arg_min[0], [2])
        self.assertEqual(arg_min[1], [0])


if __name__ == "__main__":
    tf.test.main()
