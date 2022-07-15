import math
import sys
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf

from tensorflow_similarity import types, utils


def create_lookups():
    lookups = []
    for i in range(2):
        lookup_set = []
        for j in range(2):
            label = j + i * 2
            lookup_set.append(types.Lookup(rank=j, distance=float(j), label=label))

        lookups.append(lookup_set)

    return lookups


class UtilsTest(tf.test.TestCase):
    def setUp(self):
        self.lookups = create_lookups()

    def test_tf_cap_memory_gpu_exists(self):
        tf.config.experimental.list_physical_devices = MagicMock(return_value=["foo"])
        tf.config.experimental.set_memory_growth = MagicMock()

        utils.tf_cap_memory()
        tf.config.experimental.set_memory_growth.assert_called_with("foo", True)

    def test_tf_cap_memory_no_gpu(self):
        tf.config.experimental.list_physical_devices = MagicMock(return_value=[])
        tf.config.experimental.set_memory_growth = MagicMock()

        utils.tf_cap_memory()
        tf.config.experimental.set_memory_growth.assert_not_called()

    def test_tf_cap_memory_runtime_error(self):
        tf.config.experimental.list_physical_devices = MagicMock(return_value=["foo"])
        tf.config.experimental.set_memory_growth = MagicMock(side_effect=RuntimeError("bar"))

        with self.captureWritesToStream(sys.stdout) as captured:
            utils.tf_cap_memory()

        self.assertRegex(captured.contents(), "bar")

    def test_is_tensor(self):
        self.assertTrue(utils.is_tensor_or_variable(tf.constant([0])))

    def test_is_variable(self):
        self.assertTrue(utils.is_tensor_or_variable(tf.Variable(1.0)))

    def test_is_not_tensor_or_variable(self):
        self.assertFalse(utils.is_tensor_or_variable([0]))

    def test_unpack_lookup_labels(self):
        with self.captureWritesToStream(sys.stdout) as captured:
            unpacked = utils.unpack_lookup_labels(self.lookups, dtype="int32")

        expected = tf.constant([[0, 1], [2, 3]], dtype="int32")

        self.assertAllEqual(unpacked, expected)
        self.assertEqual(captured.contents(), "")

    def test_unpack_lookup_labels_uneven_lookup_sets(self):
        # Add an extra label to the second lookup set
        self.lookups[1].append(types.Lookup(rank=3, distance=math.inf, label=4))

        with self.captureWritesToStream(sys.stdout) as captured:
            unpacked = utils.unpack_lookup_labels(self.lookups, dtype="int32")

        expected = tf.constant([[0, 1, 0x7FFFFFFF], [2, 3, 4]], dtype="int32")

        self.assertAllEqual(unpacked, expected)

        msg = (
            "WARNING: 1 lookup sets are shorter than the max lookup set "
            "length. Imputing 0x7FFFFFFF for the missing label lookups.\n"
        )

        self.assertEqual(captured.contents(), msg)

    def test_unpack_lookup_distances(self):
        with self.captureWritesToStream(sys.stdout) as captured:
            unpacked = utils.unpack_lookup_distances(self.lookups, dtype="float32")

        expected = tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype="float32")

        self.assertAllEqual(unpacked, expected)
        self.assertEqual(captured.contents(), "")

    def test_unpack_lookup_distances_rounding(self):
        lookups = [[types.Lookup(rank=0, distance=0.11119, label=1)]]

        with self.captureWritesToStream(sys.stdout) as captured:
            unpacked = utils.unpack_lookup_distances(lookups, dtype="float32", distance_rounding=4)

        expected = tf.constant([[0.1112]], dtype="float32")

        self.assertAllEqual(unpacked, expected)
        self.assertEqual(captured.contents(), "")

    def test_unpack_lookup_distances_uneven_lookup_sets(self):
        # Add an extra label to the second lookup set
        self.lookups[1].append(types.Lookup(rank=3, distance=2.0, label=4))

        with self.captureWritesToStream(sys.stdout) as captured:
            unpacked = utils.unpack_lookup_distances(self.lookups, dtype="float32")

        expected = tf.constant([[0.0, 1.0, math.inf], [0.0, 1.0, 2.0]], dtype="float32")

        self.assertAllEqual(unpacked, expected)

        msg = (
            "WARNING: 1 lookup sets are shorter than the max lookup set "
            "length. Imputing math.inf for the missing distance lookups.\n"
        )

        self.assertEqual(captured.contents(), msg)

    def test_same_length_rows_check_same_length(self):
        x = tf.ragged.constant([[0, 1], [0, 2], [0, 3]])
        self.assertTrue(utils._same_length_rows(x))

    def test_same_length_rows_check_different_lengths(self):
        x = tf.ragged.constant([[0], [0, 2], [0, 2, 3]])
        self.assertFalse(utils._same_length_rows(x))

    def test_count_of_small_lookup_sets_ragged(self):
        x = tf.ragged.constant([[0], [0, 2], [0, 2, 3]])
        counts = utils._count_of_small_lookup_sets(x)
        self.assertEqual(counts, 2)

    def test_count_of_small_lookup_sets_all_same_length(self):
        x = tf.ragged.constant([[0, 2], [0, 2], [0, 2]])
        counts = utils._count_of_small_lookup_sets(x)
        self.assertEqual(counts, 0)

    def test_unpack_results(self):
        results = {"recall@1": np.array([1.0])}
        logs = {"some_other_callback": 0.0}
        tb_writer = MagicMock()
        tf.summary.scalar = MagicMock()

        mstr = utils.unpack_results(
            results=results,
            epoch=1,
            logs=logs,
            tb_writer=tb_writer,
            name_suffix="_test",
        )

        self.assertDictEqual(logs, {"some_other_callback": 0.0, "recall@1_test": 1.0})
        self.assertListEqual(mstr, ["recall@1_test: 1.0000"])
        tf.summary.scalar.assert_called_with("recall@1_test", 1.0, step=1)


if __name__ == "__main__":
    tf.test.main()
