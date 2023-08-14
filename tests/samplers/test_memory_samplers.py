import re

import pytest
import tensorflow as tf

from tensorflow_similarity.samplers import MultiShotMemorySampler, select_examples


def test_valid_class_numbers():
    "Check that sampler properly detect if num_class requests >> class avail"
    y = tf.constant([1, 2, 3, 1, 2, 3, 1])
    x = tf.constant([10, 20, 30, 10, 20, 30, 10])

    class_per_batch = 42

    with pytest.raises(ValueError):
        MultiShotMemorySampler(x=x, y=y, classes_per_batch=class_per_batch)


@pytest.mark.parametrize("example_per_class", [2, 20])
def test_select_examples(example_per_class):
    """Test select_examples with various sizes.

    Users may sample with replacement when creating batches, so check that we
    can handle when elements per class is either less than or greater than the
    total count of elements in the class.
    """
    y = tf.constant([1, 2, 3, 1, 2, 3, 1])
    x = tf.constant([10, 20, 30, 10, 20, 30, 10])
    cls_list = [1, 3]
    batch_x, batch_y = select_examples(x, y, cls_list, example_per_class)

    assert len(batch_y) == len(cls_list) * example_per_class
    assert len(batch_x) == len(cls_list) * example_per_class

    for x, y in zip(batch_x, batch_y):
        assert y in cls_list

        if y == 1:
            assert x == 10
        elif y == 3:
            assert x == 30


@pytest.mark.parametrize("example_per_class", [2, 20])
def test_multi_shot_memory_sampler(example_per_class):
    """Test MultiShotMemorySampler with various sizes.

    Users may sample with replacement when creating batches, so check that we
    can handle when elements per class is either less than or greater than the
    total count of elements in the class.
    """
    y = tf.constant([1, 2, 3, 1, 2, 3, 1])
    x = tf.constant([10, 20, 30, 10, 20, 30, 10])
    class_per_batch = 2
    batch_size = example_per_class * class_per_batch

    ms_sampler = MultiShotMemorySampler(
        x=x,
        y=y,
        classes_per_batch=class_per_batch,
        examples_per_class_per_batch=example_per_class,
    )  # noqa

    batch_x, batch_y = ms_sampler.generate_batch(batch_id=606)

    assert len(batch_y) == batch_size
    assert len(batch_x) == batch_size
    num_classes, _ = tf.unique(batch_y)
    assert len(num_classes) == class_per_batch

    for x, y in zip(batch_x, batch_y):
        if y == 1:
            assert x == 10
        elif y == 2:
            assert x == 20
        elif y == 3:
            assert x == 30


def test_msms_get_slice():
    """Test the multi shot memory sampler get_slice method."""
    y = tf.constant(range(4))
    x = tf.constant([[0] * 10, [1] * 10, [2] * 10, [3] * 10])

    ms_sampler = MultiShotMemorySampler(x=x, y=y)
    # x and y are randomly shuffled so we fix the values here.
    ms_sampler._x = x
    ms_sampler._y = y
    slice_x, slice_y = ms_sampler.get_slice(1, 2)

    assert slice_x.shape == (2, 10)
    assert slice_y.shape == (2,)

    assert slice_x[0, 0] == 1
    assert slice_x[1, 0] == 2

    assert slice_y[0] == 1
    assert slice_y[1] == 2


def test_msms_properties():
    """Test the multi shot memory sampler num_examples and shape"""
    y = tf.constant(range(4))
    x = tf.ones([4, 10, 20, 3])

    ms_sampler = MultiShotMemorySampler(x=x, y=y)

    assert ms_sampler.num_examples == 4
    assert ms_sampler.example_shape == (10, 20, 3)


def test_small_class_size(capsys):
    """Test examples_per_class is > the number of class examples."""
    y = tf.constant([1, 1, 1, 2])
    x = tf.ones([4, 10, 10, 3])

    ms_sampler = MultiShotMemorySampler(x=x, y=y, classes_per_batch=2, examples_per_class_per_batch=3)

    _, batch_y = ms_sampler.generate_batch(0)

    y, _, class_counts = tf.unique_with_counts(batch_y)
    assert tf.math.reduce_all(tf.sort(y) == tf.constant([1, 2]))
    assert tf.math.reduce_all(class_counts == tf.constant([3, 3]))

    captured = capsys.readouterr()
    expected_msg = (
        "WARNING: Class 2 only has 1 unique examples, but "
        "examples_per_class is set to 3. The current batch will sample "
        "from class examples with replacement, but you may want to "
        "consider passing an Augmenter function or using the "
        "SingleShotMemorySampler()."
    )

    match = re.search(expected_msg, captured.out)
    assert bool(match)

    _, batch_y = ms_sampler.generate_batch(0)

    y, _, class_counts = tf.unique_with_counts(batch_y)
    assert tf.math.reduce_all(tf.sort(y) == tf.constant([1, 2]))
    assert tf.math.reduce_all(class_counts == tf.constant([3, 3]))

    # Subsequent batch should produce the sampler warning.
    captured = capsys.readouterr()
    match = re.search(expected_msg, captured.out)
    assert not bool(match)
