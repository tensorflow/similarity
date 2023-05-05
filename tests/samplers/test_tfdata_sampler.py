import tensorflow as tf

from tensorflow_similarity.samplers import tfdata_sampler as tfds


class TestFilterClasses(tf.test.TestCase):
    def setUp(self):
        self.ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                tf.constant([1, 1, 2, 2, 4, 3, 4, 3]),
            )
        )
        self.classes = {c: 0 for c in range(1, 5)}

    def test_filter_classes(self):
        ds = tfds.filter_classes(self.ds, [2, 4])
        for _, label in ds:
            self.classes[label.numpy()] += 1
        self.assertEqual(self.classes[1], 0)
        self.assertEqual(self.classes[2], 2)
        self.assertEqual(self.classes[3], 0)
        self.assertEqual(self.classes[4], 2)

    def test_filter_empty_class_list(self):
        ds = tfds.filter_classes(self.ds)
        for _, label in ds:
            self.classes[label.numpy()] += 1
        self.assertEqual(self.classes[1], 2)
        self.assertEqual(self.classes[2], 2)
        self.assertEqual(self.classes[3], 2)
        self.assertEqual(self.classes[4], 2)


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
    def setUp(self):
        self.classes = {c: 0 for c in range(5)}

    def test_choice_values_with_repetition(self):
        # Test that each class appears exactly examples_per_class * repeat times
        num_classes = 5
        examples_per_class = 2
        num_repeats = 2
        dataset = tfds.create_choices_dataset(num_classes, examples_per_class)
        for label in dataset.take(num_classes * examples_per_class * num_repeats):
            self.classes[label.numpy()] += 1

        self.assertSetEqual(set(self.classes.keys()), set(range(5)))
        self.assertEqual(self.classes[0], 4)
        self.assertEqual(self.classes[1], 4)
        self.assertEqual(self.classes[2], 4)
        self.assertEqual(self.classes[3], 4)
        self.assertEqual(self.classes[4], 4)


def dummy_augmenter(x):
    return x + 10


class TestAugmenter(tf.test.TestCase):
    def setUp(self):
        self.ds = tf.data.Dataset.range(1, 5)

    def test_apply_augmentation_no_warmup(self):
        augmented_ds = tfds.apply_augmenter_ds(self.ds, dummy_augmenter)

        for x in augmented_ds.batch(2).as_numpy_iterator():
            self.assertAllEqual(x, [11, 12])
            break

    def test_apply_augmentation_with_warmup(self):
        warmup = 2
        batch_size = 2
        augmented_ds = tfds.apply_augmenter_ds(self.ds, dummy_augmenter, warmup)

        for i, x in enumerate(augmented_ds.batch(batch_size).take(2)):
            if i < warmup // batch_size:
                self.assertAllEqual(x, tf.constant([1, 2]))
            else:
                self.assertAllEqual(x, tf.constant([13, 14]))

    def test_apply_augmentation_with_warmup_and_repeats(self):
        values = {c: 0 for c in [1, 2, 3, 4, 11, 12, 13, 14]}
        warmup = 2
        augmented_ds = tfds.apply_augmenter_ds(self.ds.repeat(2), dummy_augmenter, warmup)

        for x in augmented_ds:
            values[x.numpy()] += 1
        self.assertEqual(values[1], 1)
        self.assertEqual(values[2], 1)
        self.assertEqual(values[3], 0)
        self.assertEqual(values[4], 0)
        self.assertEqual(values[11], 1)
        self.assertEqual(values[12], 1)
        self.assertEqual(values[13], 2)
        self.assertEqual(values[14], 2)


class TestTFDataSampler(tf.test.TestCase):
    def setUp(self):
        self.ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.expand_dims(tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.int32), axis=-1),
                tf.constant([1, 1, 1, 2, 2, 2], dtype=tf.int32),
            )
        )
        self.expected_element_spec = (
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
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
        out_ds = tfds.TFDataSampler(self.ds, classes_per_batch=2, examples_per_class_per_batch=2)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)
        for _, y in out_ds.take(1):
            self.assertLen(y, 4)

    def test_output_classes_per_batch(self):
        # Test that the number of classes per batch is correct
        out_ds = tfds.TFDataSampler(self.ds, classes_per_batch=1)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)
        for _, y in out_ds.take(2):
            self.assertLen(tf.unique(y).y, 1)

    def test_output_examples_per_class_per_batch(self):
        # Test that the number of examples per class per batch is correct
        out_ds = tfds.TFDataSampler(self.ds, classes_per_batch=2, examples_per_class_per_batch=3)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)
        for _, y in out_ds.take(2):
            self.assertAllEqual(tf.math.bincount(y), [0, 3, 3])

    def test_output_class_list(self):
        # Test that the class list is correctly used
        out_ds = tfds.TFDataSampler(self.ds, class_list=[1])
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)
        for _, y in out_ds.take(4):
            self.assertAllEqual(tf.unique(y).y, [1])

    def test_output_total_examples_per_class(self):
        # Test that the total number of examples per class is correctly used
        out_ds = tfds.TFDataSampler(self.ds, classes_per_batch=2, total_examples_per_class=2)
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)
        for x, y in out_ds.take(2):
            for x_, y_ in zip(x, y):
                if y_ == 1:
                    self.assertLessEqual(x_, 2)
                elif y_ == 2:
                    self.assertGreater(x_, 2)
                else:
                    raise ValueError("Unexpected class")

    def test_output_augmenter_with_warmup_and_repeat(self):
        # Test that the augmenter is correctly applied
        out_ds = tfds.TFDataSampler(
            self.ds,
            classes_per_batch=2,
            examples_per_class_per_batch=3,
            augmenter=lambda x, y: (x + 6, y),
            warmup=6,
        )
        self.assertEqual(out_ds.element_spec, self.expected_element_spec)
        values = {c: 0 for c in range(12)}
        for x, y in out_ds.take(3):
            for x_ in x:
                values[x_.numpy()[0]] += 1
        self.assertEqual(values[1], 1)
        self.assertEqual(values[2], 1)
        self.assertEqual(values[3], 1)
        self.assertEqual(values[4], 1)
        self.assertEqual(values[5], 1)
        self.assertEqual(values[6], 2)
        self.assertEqual(values[7], 2)
        self.assertEqual(values[8], 2)
        self.assertEqual(values[9], 2)
        self.assertEqual(values[10], 2)
        self.assertEqual(values[11], 2)

    def test_output_load_fn(self):
        out_ds = tfds.TFDataSampler(
            self.ds,
            classes_per_batch=2,
            total_examples_per_class=2,
            load_fn=lambda x, y, *args: ("loaded_img", y, *args),
        )
        for x, y in out_ds:
            self.assertAllEqual(x.numpy(), tf.constant([b"loaded_img"] * 4))
            break

    def test_label_output_dict(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                {
                    "A": tf.constant([1, 1, 1, 1, 1, 1, 1, 1]),
                    "B": tf.constant([3, 3, 3, 3, 2, 2, 2, 2]),
                },
            )
        )
        out_ds = tfds.TFDataSampler(
            ds,
            classes_per_batch=2,
            examples_per_class_per_batch=3,
            label_output="B",
        )
        for _, y in out_ds.take(4):
            self.assertSetEqual(set(tf.unique(y["B"]).y.numpy()), set([2, 3]))
            self.assertAllEqual(tf.math.bincount(y["B"]), [0, 0, 3, 3])

    def test_label_output_tuple(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                (
                    tf.constant([1, 1, 1, 1, 1, 1, 1, 1]),
                    tf.constant([3, 3, 3, 3, 2, 2, 2, 2]),
                ),
            )
        )
        out_ds = tfds.TFDataSampler(
            ds,
            classes_per_batch=2,
            examples_per_class_per_batch=3,
            label_output=1,
        )
        for _, y in out_ds.take(4):
            self.assertSetEqual(set(tf.unique(y[1]).y.numpy()), set([2, 3]))
            self.assertAllEqual(tf.math.bincount(y[1]), [0, 0, 3, 3])


if __name__ == "__main__":
    tf.test.main()
