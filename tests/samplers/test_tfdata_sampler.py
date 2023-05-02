from collections import defaultdict

import tensorflow as tf

from tensorflow_similarity.samplers import tfdata_sampler as tfds


class TestCreateGroupedDataset(tf.test.TestCase):
    def setUp(self):
        self.ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                tf.constant([1, 1, 2, 2, 3, 3, 4, 4]),
            )
        )
        self.window_size = self.ds.cardinality().numpy()

    def test_returns_correct_number_of_datasets(self):
        cid_datasets = tfds.create_grouped_dataset(self.ds, self.window_size)
        self.assertLen(cid_datasets, 4)

    def test_returns_correct_number_of_datasets_with_total_examples(self):
        cid_datasets = tfds.create_grouped_dataset(self.ds, self.window_size, total_examples=1)
        self.assertLen(cid_datasets, 4)

        # test that each cid dataset has only 1 example that is repeated.
        for elem in cid_datasets:
            self.assertAllEqual(
                list(elem.take(1).as_numpy_iterator()),
                list(elem.take(1).as_numpy_iterator()),
            )

    def test_returns_correct_number_of_datasets_with_buffer_size(self):
        cid_datasets = tfds.create_grouped_dataset(self.ds, self.window_size, buffer_size=2)
        self.assertEqual(len(cid_datasets), 4)

    def test_datasets_repeat(self):
        print(self.ds.element_spec)
        cid_datasets = tfds.create_grouped_dataset(self.ds, self.window_size)
        for cid_ds in cid_datasets:
            self.assertTrue(cid_ds.element_spec, self.ds.element_spec)

            # Check that repeating groups of 2 elements all equal each other.
            elements = list(cid_ds.take(6).as_numpy_iterator())
            self.assertAllEqual(elements[:2], elements[2:4])
            self.assertAllEqual(elements[:2], elements[4:])

    def test_datasets_shuffled(self):
        cid_datasets = tfds.create_grouped_dataset(self.ds, self.window_size, buffer_size=4)
        # check that including the buffer shuffles the values in each cid_ds.
        for cid_ds in cid_datasets:
            self.assertNotEqual(
                list(cid_ds.take(20).as_numpy_iterator()),
                list(cid_ds.take(20).as_numpy_iterator()),
            )


class TestCreateChoicesDataset(tf.test.TestCase):
    def test_sample_without_replacement(self):
        # Test that each class appears exactly examples_per_class times
        num_classes = 5
        examples_per_class = 2
        dataset = tfds.create_choices_dataset(num_classes, examples_per_class)
        elements = list(dataset.take(num_classes * examples_per_class).as_numpy_iterator())
        unique_elements = set(elements)
        self.assertLen(unique_elements, num_classes)

    def test_dataset_values(self):
        # Test that the dataset only contains values between 0 and num_classes
        num_classes = 10
        examples_per_class = 3
        dataset = tfds.create_choices_dataset(num_classes, examples_per_class)
        for x in dataset.take(num_classes * examples_per_class).as_numpy_iterator():
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, num_classes)

    def test_dataset_repetition(self):
        # Test that each class appears exactly examples_per_class times
        num_classes = 4
        examples_per_class = 2
        num_repeats = 2
        dataset = tfds.create_choices_dataset(num_classes, examples_per_class)
        class_counts = defaultdict(int)
        for x in dataset.take(num_classes * examples_per_class * num_repeats).as_numpy_iterator():
            class_counts[x] += 1
        for count in class_counts.values():
            self.assertEqual(count, examples_per_class * num_repeats)


def dummy_augmenter(x):
    return x + 10


class TestAugmenter(tf.test.TestCase):
    def setUp(self):
        self.ds = tf.data.Dataset.range(10).batch(2)

    def test_apply_augmentation_no_warmup(self):
        augmented_ds = tfds.apply_augmenter_ds(self.ds, dummy_augmenter)

        for x in augmented_ds:
            self.assertListEqual(x.numpy().tolist(), [10, 11])
            break

    def test_apply_augmentation_with_warmup(self):
        warmup = 1
        augmented_ds = tfds.apply_augmenter_ds(self.ds, dummy_augmenter, warmup)

        for i, x in enumerate(augmented_ds):
            if i < warmup:
                self.assertListEqual(x.numpy().tolist(), [0, 1])
            else:
                self.assertListEqual(x.numpy().tolist(), [12, 13])
                break


class TestTFDataSampler(tf.test.TestCase):
    def setUp(self):
        self.ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.random.uniform((6, 2), dtype=tf.float32),
                tf.constant([1, 1, 1, 2, 2, 2], dtype=tf.int32),
            )
        )
        self.expected_element_spec = (
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

    def test_cardinality_is_finite(self):
        # Test that an exception is raised when the input dataset is infinite
        ds = tf.data.Dataset.from_tensors([1]).repeat()
        with self.assertRaisesWithLiteralMatch(ValueError, "Dataset must be finite."):
            tfds.TFDataSampler(ds)

    def test_cardinality_is_known(self):
        # Test that an exception is raised when the input dataset has unknown cardinality
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3]).shuffle(3).filter(lambda x: x > 1)
        with self.assertRaisesWithLiteralMatch(ValueError, "Dataset cardinality must be known."):
            tfds.TFDataSampler(ds)

    def test_output_batch_size(self):
        # Test that the output batch size is correct
        out_ds = tfds.TFDataSampler(self.ds)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)

    def test_output_classes_per_batch(self):
        # Test that the number of classes per batch is correct
        out_ds = tfds.TFDataSampler(self.ds, classes_per_batch=1)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)

    def test_output_examples_per_class_per_batch(self):
        # Test that the number of examples per class per batch is correct
        out_ds = tfds.TFDataSampler(self.ds, examples_per_class_per_batch=1)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)

    def test_output_class_list(self):
        # Test that the class list is correctly used
        out_ds = tfds.TFDataSampler(self.ds, class_list=[1])
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)

    def test_output_total_examples_per_class(self):
        # Test that the total number of examples per class is correctly used
        out_ds = tfds.TFDataSampler(self.ds, total_examples_per_class=2)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)

    def test_output_augmenter(self):
        # Test that the augmenter is correctly applied
        out_ds = tfds.TFDataSampler(self.ds, augmenter=lambda x, y: (x * 2, y))
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)

    def test_output_load_fn(self):
        # TODO(ovallis): Test that the load_fn is correctly applied
        pass


if __name__ == "__main__":
    tf.test.main()
