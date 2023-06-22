from __future__ import annotations

from collections.abc import Callable, Sequence

import tensorflow as tf


def filter_classes(
    ds: tf.data.Dataset,
    class_list: Sequence[int] | None = None,
    y_parser: Callable = lambda y: y,
) -> tf.data.Dataset:
    """
    Filters a dataset by class id.

    Args:
        ds: A `tf.data.Dataset` object.
        class_list: An optional `Sequence` of integers representing the classes
            to include in the dataset. If `None`, all classes are included.
        y_parser: A callable function used to parse the class id from the y outputs.

    Returns:
        A `tf.data.Dataset` object filtered by class id.
    """

    if class_list is not None:
        class_list = tf.constant(class_list)
        ds = ds.filter(lambda x, y, *args: tf.reduce_any(tf.equal(y_parser(y), class_list)))

    return ds


def create_grouped_dataset(
    ds: tf.data.Dataset,
    window_size: int,
    total_examples: int | None = None,
    buffer_size: int | None = None,
    y_parser: Callable = lambda y: y,
) -> list[tf.data.Dataset]:
    """
    Creates a list of datasets grouped by class id.

    Args:
        ds: A `tf.data.Dataset` object.
        window_size: An integer representing the dataset cardinality.
        total_examples: An integer representing the maximum number of examples
            to include in the dataset. If `None`, all examples are included.
        buffer_size: An optional integer representing the size of the buffer
            for shuffling. Default is None.
        y_parser: A callable function used to parse the class id from the y outputs.

    Returns:
        A List of `tf.data.Dataset` objects grouped by class id.
    """
    # NOTE: We need to cast the key_func as the group_op expects an int64.
    grouped_by_cid = ds.group_by_window(
        key_func=lambda x, y, *args: tf.cast(y_parser(y), dtype=tf.int64),
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


def apply_augmenter_ds(ds: tf.data.Dataset, augmenter: Callable, warmup: int = 0) -> tf.data.Dataset:
    """
    Applies an augmenter function to a dataset batch and optionally delays
    applying the function for `warmup` number of examples.

    Args:
        ds: A `tf.data.Dataset` object.
        augmenter: A callable function used to apply data augmentation to
            individual examples. If `None`, no data augmentation is applied.
        warmup: An integer representing the number of examples to wait
            before applying the data augmentation function.

    Returns:
        A `tf.data.Dataset` object with the applied augmenter.
    """
    if not warmup:
        return ds.map(augmenter, name="augmenter")

    aug_ds = ds.map(augmenter, name="augmenter").skip(warmup)

    try:
        count_ds = tf.data.Dataset.counter()
    except AttributeError:
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
    warmup: int = 0,
    label_output: int | str | None = None,
) -> tf.data.Dataset:
    """
    Returns a `tf.data.Dataset` object that generates batches of examples with an
    equal number of examples per class. The input dataset cardinality must be finite
    and known.

    Args:
        ds: A `tf.data.Dataset` object representing the original dataset.
        classes_per_batch: An integer specifying the number of classes per batch.
        examples_per_class_per_batch: An integer specifying the number of examples
            per class per batch.
        class_list: An optional sequence of integers representing the class IDs to
            include in the dataset. If `None`, all classes in the original dataset
            will be used.
        total_examples_per_class: An optional integer representing the total number
            of examples per class to use. If `None`, all examples for each class will
            be used.
        augmenter: An optional function to apply data augmentation to each example.
        load_fn: An optional callable function for loading real examples from `x`. It
            is useful for loading images from their corresponding file path provided
            in `x` or similar situations.
        warmup: An integer specifying the number of examples to use for unaugmented
            warmup.
        label_output: An optional integer or string representing the label output used
            to create the balanced dataset batches. If `None`, y is assumed to be a 1D
            integer tensor containing the class IDs.

    Returns:
        A `tf.data.Dataset` object representing the balanced dataset batches.

    Raises:
        ValueError: If `ds` is an infinite dataset or the cardinality is unknown.
    """
    if ds.cardinality() == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Dataset must be finite.")
    if ds.cardinality() == tf.data.UNKNOWN_CARDINALITY:
        raise ValueError("Dataset cardinality must be known.")

    def y_parser(y):
        return y if label_output is None else y[label_output]

    window_size = ds.cardinality().numpy()
    batch_size = examples_per_class_per_batch * classes_per_batch

    ds = filter_classes(ds, class_list=class_list, y_parser=y_parser)
    ds = create_grouped_dataset(
        ds,
        window_size=window_size,
        total_examples=total_examples_per_class,
        y_parser=y_parser,
    )
    choices_ds = create_choices_dataset(
        len(ds),
        examples_per_class=examples_per_class_per_batch,
    )
    ds = tf.data.Dataset.choose_from_datasets(ds, choices_ds)

    if load_fn is not None:
        ds = ds.map(load_fn, name="load_fn")

    if augmenter is not None:
        ds = apply_augmenter_ds(ds, augmenter, warmup)

    ds = ds.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
