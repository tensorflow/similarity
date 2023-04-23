from __future__ import annotations

import tensorflow as tf


def TFDataSampler(
    dataset: tf.data.Dataset,
    load_example_fn: Callable | None = None,
    classes_per_batch: int = 2,
    examples_per_class_per_batch: int = 2,
    class_list: Sequence[int] | None = None,
    total_examples_per_class: int | None = None,
    augmenter: Augmenter | None = None,
    warmup: int = -1,
):
  """Create a tf.data.Dataset sampler that ensures that each batch is

  well balanced. That is, each batch aims to contain
  `examples_per_class_per_batch` examples of `classes_per_batch` classes.
  The `batch_size` used during training will be equal to:
  `classes_per_batch` * `examples_per_class_per_batch` unless an
  `augmenter` that alters the number of examples returned is used. Then
  the batch_size is a function of how many augmented examples are
  returned by the `augmenter`.
  This sampler is to be used when you have multiple examples for
  the same class. If this is not the case, then see the
  [SingleShotMemorySampler()](single_memory.md) for using single examples
  with augmentation.
  Args:
      dataset: A tf.data.Dataset containing at least values for x and y.
      load_example_fn: A function for loading real examples from `x`. It is
        useful for loading images from their corresponding file path provided in
        `x` or similar situations.
      classes_per_batch: Numbers of distinct class to include in a single batch
      examples_per_class_per_batch: How many example of each class to use per
        batch. Defaults to 2.
      class_list: Filter the list of examples to only keep those who belong to
        the supplied class list.
      total_examples_per_class: Restrict the number of examples for EACH class
        to total_examples_per_class if set. If not set, all the available
        examples are selected. Defaults to None - no selection.
      augmenter: A function that takes a batch in and return a batch out. Can
        alters the number of examples returned which in turn change the
        batch_size used. Defaults to None.
      warmup: Keep track of warmup epochs and let the augmenter knows when the
        warmup is over by passing along with each batch data a boolean
        `is_warmup`. See `self._get_examples()` Defaults to 0.
  """
  window_size = dataset.cardinality().numpy()
  group_by_cid = dataset.group_by_window(
      key_func=_key_func,
      reduce_func=lambda key, ds: ds.batch(window_size),
      window_size=window_size,
  )

  cid_datasets = []
  for elem in group_by_cid.as_numpy_iterator():
    if class_list is not None and elem[1][0] not in class_list:
      continue
    if total_examples_per_class is not None:
      elem = elem[:total_examples_per_class]
    cid_datasets.append(
        tf.data.Dataset.from_tensor_slices(elem)
        .shuffle(100)
        .repeat()
    )
  choices_ds = _build_choices_ds(
      len(cid_datasets), examples_per_class_per_batch
  )

  batch_size = examples_per_class_per_batch * classes_per_batch
  ds = (
      tf.data.Dataset.choose_from_datasets(cid_datasets, choices_ds)
      .repeat()
      .batch(batch_size)
  )

  if load_example_fn is not None:
    ds = ds.map(load_example_fn, name="load_example_fn")

  if augmenter is not None:
    ds = _build_augmenter_ds(ds, augmenter, warmup)

  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def _key_func(_, y, *args):
  return y


def _build_choices_ds(num_classes, examples_per_class_per_batch):
  return (
      tf.data.Dataset.range(num_classes)
      .shuffle(num_classes)
      .map(lambda z: [[z] * examples_per_class_per_batch], name="repeat_cid")
      .flat_map(tf.data.Dataset.from_tensor_slices)
  )


def _build_augmenter_ds(ds, augmenter, warmup):
  if warmup == -1:
    return ds

  ds = tf.data.Dataset.choose_from_datasets(
      [ds, ds.map(augmenter, name="augmenter")],
      tf.data.Counter().map(lambda x: 0 if x < warmup else 1),
  )

  return ds
