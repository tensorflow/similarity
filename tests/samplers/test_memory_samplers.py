import tensorflow as tf
from tensorflow_similarity.samplers import select_examples


def test_select_examples():
    y = tf.constant([1, 2, 3, 1, 2, 3])
    x = tf.constant([10, 20, 30, 10, 20, 30])
    cls_list = [1, 3]
    elt_per_class = 2
    batch_x, batch_y = select_examples(x, y, cls_list, elt_per_class)

    assert len(batch_y) == len(cls_list) * elt_per_class
    assert len(batch_x) == len(cls_list) * elt_per_class

    for e in batch_y:
        assert e == 1 or e == 3

    for e in batch_x:
        assert e == 10 or e == 30
