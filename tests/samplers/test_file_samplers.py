import os
import re
import tempfile

import numpy as np

import pytest
import tensorflow as tf

from tensorflow_similarity.samplers import MultiShotFileSampler


def _create_random_image(filename, size=(32, 32)):
    filepath = os.path.join(tempfile.gettempdir(), filename)
    image = np.random.random(size + (3,)).astype(np.float32)
    tf.keras.utils.save_img(filepath, image)
    return filepath


@pytest.mark.parametrize("example_per_class", [2, 20])
def test_multi_shot_file_sampler(example_per_class):
    """Test MultiShotFileSampler with various sizes.

    Users may sample with replacement when creating batches, so check that we
    can handle when elements per class is either less than or greater than the
    total count of elements in the class.
    """
    filenames = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg']
    filepaths = [
        _create_random_image(filename) for filename in filenames
    ]
    images = [
        np.array(tf.keras.utils.load_img(path), dtype=np.float32) / 255
        for path in filepaths
    ]

    y = tf.constant([1, 2, 3, 1, 2, 3])
    x = tf.constant(filepaths)
    class_per_batch = 2
    batch_size = example_per_class * class_per_batch

    fs_sampler = MultiShotFileSampler(
        x=x,
        y=y,
        classes_per_batch=class_per_batch,
        examples_per_class_per_batch=example_per_class,
    )  # noqa

    batch_x, batch_y = fs_sampler.generate_batch(batch_id=606)

    assert len(batch_y) == batch_size
    assert len(batch_x) == batch_size
    num_classes, _ = tf.unique(batch_y)
    assert len(num_classes) == class_per_batch

    for x, y in zip(batch_x, batch_y):
        if y == 1:
            assert (
                np.isclose(x.numpy(), images[0], atol=.1).all() or
                np.isclose(x.numpy(), images[3], atol=.1).all()
            )
        elif y == 2:
            assert (
                np.isclose(x.numpy(), images[1], atol=.1).all() or
                np.isclose(x.numpy(), images[4], atol=.1).all()
            )
        elif y == 3:
            assert (
                np.isclose(x.numpy(), images[2], atol=.1).all() or
                np.isclose(x.numpy(), images[5], atol=.1).all()
            )


def test_msfs_get_slice():
    """Test the multi shot file sampler get_slice method."""
    filenames = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    filepaths = [
        _create_random_image(filename) for filename in filenames
    ]
    images = [
        np.array(tf.keras.utils.load_img(path), dtype=np.float32) / 255
        for path in filepaths
    ]

    y = tf.constant(range(4))
    x = tf.constant(filepaths)

    fs_sampler = MultiShotFileSampler(x=x, y=y)
    # x and y are randomly shuffled so we fix the values here.
    fs_sampler._x = x
    fs_sampler._y = y
    slice_x, slice_y = fs_sampler.get_slice(1, 2)

    assert slice_x.shape == (2, 32, 32, 3)
    assert slice_y.shape == (2,)

    assert np.isclose(slice_x[0], images[1], atol=.1).all()
    assert np.isclose(slice_x[1], images[2], atol=.1).all()

    assert slice_y[0] == 1
    assert slice_y[1] == 2


def test_msms_properties():
    """Test the multi shot file sampler num_examples and shape"""
    filenames = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    filepaths = [
        _create_random_image(filename, (128, 96)) for filename in filenames
    ]
    y = tf.constant(range(4))
    x = tf.constant(filepaths)

    fs_sampler = MultiShotFileSampler(x=x, y=y)

    assert fs_sampler.num_examples == 4
    assert fs_sampler.example_shape == (128, 96, 3)


def test_small_class_size(capsys):
    """Test examples_per_class is > the number of class examples."""
    filenames = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    filepaths = [
        _create_random_image(filename) for filename in filenames
    ]

    y = tf.constant([1, 1, 1, 2])
    x = tf.constant(filepaths)

    fs_sampler = MultiShotFileSampler(x=x, y=y, classes_per_batch=2, examples_per_class_per_batch=3)

    _, batch_y = fs_sampler.generate_batch(0)

    y, _, class_counts = tf.unique_with_counts(batch_y)
    assert tf.math.reduce_all(tf.sort(y) == tf.constant([1, 2]))
    assert tf.math.reduce_all(class_counts == tf.constant([3, 3]))

    captured = capsys.readouterr()
    expected_msg = (
        "WARNING: Class 2 only has 1 unique examples, but "
        "examples_per_class is set to 3. The current batch will sample "
        "from class examples with replacement, but you may want to "
        "consider passing an Augmenter function or using the "
        "SingleShotFileSampler()."
    )

    match = re.search(expected_msg, captured.out)
    assert bool(match)

    _, batch_y = fs_sampler.generate_batch(0)

    y, _, class_counts = tf.unique_with_counts(batch_y)
    assert tf.math.reduce_all(tf.sort(y) == tf.constant([1, 2]))
    assert tf.math.reduce_all(class_counts == tf.constant([3, 3]))

    # Subsequent batch should produce the sampler warning.
    captured = capsys.readouterr()
    match = re.search(expected_msg, captured.out)
    assert not bool(match)
