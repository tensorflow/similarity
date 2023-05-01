from __future__ import annotations

from collections.abc import Callable, Sequence

import tensorflow as tf


def create_grouped_dataset(
    ds: tf.data.Dataset,
    class_list: Sequence[int] | None = None,
    total_examples: int | None = None,
    buffer_size: int | None = None,
) -> list[tf.data.Dataset]:
    """
    Creates a list of datasets grouped by class id.

    Args:
        ds: A `tf.data.Dataset` object.
        class_list: An optional `Sequence` of integers representing the classes
            to include in the dataset. If `None`, all classes are included.
        total_examples: An integer representing the maximum number of examples
            to include in the dataset. If `None`, all examples are included.
        buffer_size: An optional integer representing the size of the buffer
            for shuffling. Default is None.

    Returns:
        A List of `tf.data.Dataset` objects grouped by class id.
    """
    window_size = ds.cardinality().numpy()
    if class_list is not None:
        class_list = tf.constant(class_list)
        ds = ds.filter(lambda x, y, *args: tf.reduce_any(tf.equal(y, class_list)))

    # NOTE: We need to cast the key_func as the group_op expects an int64.
    grouped_by_cid = ds.group_by_window(
        key_func=lambda x, y, *args: tf.cast(y, dtype=tf.int64),
        reduce_func=lambda key, ds: ds.batch(window_size),
        window_size=window_size,
    )

    cid_datasets = []
    for elem in grouped_by_cid:
        cid_ds = tf.data.Dataset.from_tensor_slices(elem)
        if total_examples is not None:
            cid_ds = cid_ds.take(total_examples)
        if buffer_size is not None:
            cid_ds = cid_ds.shuffle(buffer_size)
        cid_datasets.append(cid_ds.repeat())

    return cid_datasets


def create_choices_dataset(num_classes: int, examples_per_class: int) -> tf.data.Dataset:
    """
    Creates a dataset that generates random integers between 0 and `num_classes`.
    Integers will be generated in contiguous blocks of size `examples_per_class`.
    Integers are sampled without replacement and are not selected again until all
    other interger values have been sampled.

    Args:
        num_classes: An integer representing the total number of classes in the dataset.
        examples_per_class: An integer representing the number of examples per class.

    Returns:
        A `tf.data.Dataset` object representing the dataset with random choices.
    """
    return (
        tf.data.Dataset.range(num_classes)
        .shuffle(num_classes)
        .map(lambda z: [[z] * examples_per_class], name="repeat_cid")
        .flat_map(tf.data.Dataset.from_tensor_slices)
        .repeat()
    )


def apply_augmenter_ds(ds: tf.data.Dataset, augmenter: Callable, warmup: int | None = None) -> tf.data.Dataset:
    """
    Applies an augmenter function to a dataset batch and optionally delays
    applying the function for `warmup` number of batches.

    Args:
        ds: A `tf.data.Dataset` object.
        augmenter: A callable function used to apply data augmentation to
            individual examples within each batch. If `None`, no data
            augmentation is applied.
        warmup: An optional integer representing the number of batches to wait
            before applying the data augmentation function. If `None`, no
            warmup is applied.

    Returns:
        A `tf.data.Dataset` object with the applied augmenter.
    """
    if warmup is None:
        return ds.map(augmenter, name="augmenter")

    aug_ds = ds.map(augmenter, name="augmenter").skip(warmup)
    tf_version_split = tf.__version__.split(".")
    if int(tf_version_split[0]) >= 2 and int(tf_version_split[1]) >= 10:
        count_ds = tf.data.Dataset.counter()
    else:
        count_ds = tf.data.experimental.Counter()

    ds = tf.data.Dataset.choose_from_datasets(
        [ds.take(warmup), aug_ds],
        count_ds.map(lambda x: tf.cast(0, dtype=tf.int64) if x < warmup else tf.cast(1, dtype=tf.int64)),
    )

    return ds


def TFDataSampler(
    ds: tf.data.Dataset,
    classes_per_batch: int = 2,
    examples_per_class_per_batch: int = 2,
    class_list: Sequence[int] | None = None,
    total_examples_per_class: int | None = None,
    augmenter: Callable | None = None,
    load_fn: Callable | None = None,
    warmup: int | None = None,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` object that generates batches of examples with
    equal number of examples per class. The input dataset cardinality must be
    finite and known.

    Args:
        ds: A `tf.data.Dataset` object representing the original dataset.
        classes_per_batch: An integer specifying the number of classes per batch.
        examples_per_class_per_batch: An integer specifying the number of examples
            per class per batch.
        class_list: An optional sequence of integers representing the class IDs
            to include in the dataset. If `None`, all classes in the original
            dataset will be used.
        total_examples_per_class: An optional integer representing the total
            number of examples per class to use.  If `None`, all examples for
            each class will be used.
        augmenter: An optional function to apply data augmentation to each
            example in a batch.
        load_fn: An optional callable function that loads examples from disk.
        warmup: An optional integer specifying the number of batches to use for
            unaugmented warmup. If `None`, no warmup will be used.

    Returns:
        A `tf.data.Dataset` object representing the balanced dataset for few-shot learning tasks.
    """
    if ds.cardinality() == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Dataset must be finite.")
    if ds.cardinality() == tf.data.UNKNOWN_CARDINALITY:
        raise ValueError("Dataset cardinality must be known.")

    grouped_dataset = create_grouped_dataset(ds, class_list, total_examples_per_class)
    choices_ds = create_choices_dataset(len(grouped_dataset), examples_per_class_per_batch)

    batch_size = examples_per_class_per_batch * classes_per_batch

    ds = tf.data.Dataset.choose_from_datasets(grouped_dataset, choices_ds).repeat().batch(batch_size)

    if load_fn is not None:
        ds = ds.map(load_fn, name="load_example_fn")

    if augmenter is not None:
        ds = apply_augmenter_ds(ds, augmenter, warmup)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
