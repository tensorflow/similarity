import tensorflow as tf
from tensorflow_similarity.samplers import select_examples
from tensorflow_similarity.samplers import MultiShotMemorySampler
import pytest


def test_valid_class_numbers():
    "Check that sampler properly detect if num_class requests >> class avail"
    y = tf.constant([1, 2, 3, 1, 2, 3, 1])
    x = tf.constant([10, 20, 30, 10, 20, 30, 10])

    class_per_batch = 42

    with pytest.raises(ValueError):
        MultiShotMemorySampler(x=x, y=y,
                               classes_per_batch=class_per_batch)


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

    ms_sampler = MultiShotMemorySampler(x=x,
                                        y=y,
                                        classes_per_batch=class_per_batch,
                                        examples_per_class_per_batch=example_per_class)  # noqa

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
