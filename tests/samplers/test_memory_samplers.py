import re
import sys

import tensorflow as tf

from tensorflow_similarity.samplers import MultiShotMemorySampler, select_examples


class MemorySamplersTest(tf.test.TestCase):
    def test_valid_class_numbers(self):
        "Check that sampler properly detect if num_class requests >> class avail"
        y = tf.constant([1, 2, 3, 1, 2, 3, 1])
        x = tf.constant([10, 20, 30, 10, 20, 30, 10])

        class_per_batch = 42
        with self.assertRaises(ValueError):
            MultiShotMemorySampler(x=x, y=y, classes_per_batch=class_per_batch)

    def test_select_examples(self):
        """Test select_examples with various sizes.

        Users may sample with replacement when creating batches, so check that we
        can handle when elements per class is either less than or greater than the
        total count of elements in the class.
        """

        examples_per_class = (2, 20)

        for example_per_class in examples_per_class:
            y = tf.constant([1, 2, 3, 1, 2, 3, 1])
            x = tf.constant([10, 20, 30, 10, 20, 30, 10])
            cls_list = [1, 3]
            with self.subTest(example_per_class=example_per_class):
                batch_x, batch_y = select_examples(x, y, cls_list, example_per_class)

                self.assertLen(batch_y, len(cls_list) * example_per_class)
                self.assertLen(batch_x, len(cls_list) * example_per_class)

                for x, y in zip(batch_x, batch_y):
                    self.assertIn(y, cls_list)

                    if y == 1:
                        self.assertEqual(x, 10)
                    elif y == 3:
                        self.assertEqual(x, 30)

    def test_multi_shot_memory_sampler(self):
        """Test MultiShotMemorySampler with various sizes.

        Users may sample with replacement when creating batches, so check that we
        can handle when elements per class is either less than or greater than the
        total count of elements in the class.
        """

        examples_per_class = (2, 20)

        for example_per_class in examples_per_class:
            y = tf.constant([1, 2, 3, 1, 2, 3, 1])
            x = tf.constant([10, 20, 30, 10, 20, 30, 10])
            class_per_batch = 2
            batch_size = example_per_class * class_per_batch
            with self.subTest(example_per_class=example_per_class):
                ms_sampler = MultiShotMemorySampler(
                    x=x,
                    y=y,
                    classes_per_batch=class_per_batch,
                    examples_per_class_per_batch=example_per_class,
                )  # noqa

                batch_x, batch_y = ms_sampler.generate_batch(batch_id=606)

                self.assertLen(batch_y, batch_size)
                self.assertLen(batch_x, batch_size)
                num_classes, _ = tf.unique(batch_y)
                self.assertLen(num_classes, class_per_batch)

                for x, y in zip(batch_x, batch_y):
                    if y == 1:
                        self.assertEqual(x, 10)
                    elif y == 2:
                        self.assertEqual(x, 20)
                    elif y == 3:
                        self.assertEqual(x, 30)

    def test_msms_get_slice(self):
        """Test the multi shot memory sampler get_slice method."""
        y = tf.constant(range(4))
        x = tf.constant([[0] * 10, [1] * 10, [2] * 10, [3] * 10])

        ms_sampler = MultiShotMemorySampler(x=x, y=y)
        # x and y are randomly shuffled so we fix the values here.
        ms_sampler._x = x
        ms_sampler._y = y
        slice_x, slice_y = ms_sampler.get_slice(1, 2)

        self.assertEqual(slice_x.shape, (2, 10))
        self.assertEqual(slice_y.shape, (2,))

        self.assertEqual(slice_x[0, 0], 1)
        self.assertEqual(slice_x[1, 0], 2)

        self.assertEqual(slice_y[0], 1)
        self.assertEqual(slice_y[1], 2)

    def test_msms_properties(self):
        """Test the multi shot memory sampler num_examples and shape"""
        y = tf.constant(range(4))
        x = tf.ones([4, 10, 20, 3])

        ms_sampler = MultiShotMemorySampler(x=x, y=y)

        self.assertEqual(ms_sampler.num_examples, 4)
        self.assertEqual(ms_sampler.example_shape, (10, 20, 3))

    def test_small_class_size(self):
        """Test examples_per_class is > the number of class examples."""
        y = tf.constant([1, 1, 1, 2])
        x = tf.ones([4, 10, 10, 3])

        with self.captureWritesToStream(sys.stdout) as captured:
            ms_sampler = MultiShotMemorySampler(x=x, y=y, classes_per_batch=2, examples_per_class_per_batch=3)
            _, batch_y = ms_sampler.generate_batch(0)
            y, _, class_counts = tf.unique_with_counts(batch_y)

        self.assertAllEqual(tf.sort(y), tf.constant([1, 2]))
        self.assertAllEqual(class_counts, tf.constant([3, 3]))

        expected_msg = (
            "WARNING: Class 2 only has 1 unique examples, but "
            "examples_per_class is set to 3. The current batch will sample "
            "from class examples with replacement, but you may want to "
            "consider passing an Augmenter function or using the "
            "SingleShotMemorySampler()."
        )

        match = re.search(expected_msg, captured.contents())
        self.assertIsNotNone(match)

        with self.captureWritesToStream(sys.stdout) as captured:
            _, batch_y = ms_sampler.generate_batch(0)
            y, _, class_counts = tf.unique_with_counts(batch_y)

        self.assertAllEqual(tf.sort(y), tf.constant([1, 2]))
        self.assertAllEqual(class_counts, tf.constant([3, 3]))

        # Subsequent batch should produce the sampler warning.
        match = re.search(expected_msg, captured.contents())
        self.assertIsNone(match)
