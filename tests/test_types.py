import random

import numpy as np
import tensorflow as tf

from tensorflow_similarity import types

random.seed(303)
np.random.seed(606)
tf.random.set_seed(808)


def basic_lookup(rank=1, distance=0.1):
    return types.Lookup(rank=rank, distance=distance)


def label_lookup(label=10):
    return types.Lookup(rank=1, distance=0.1, label=label)


def embedding_lookup(embedding):
    return types.Lookup(rank=1, distance=0.1, embedding=embedding)


def data_lookup(data):
    return types.Lookup(rank=1, distance=0.1, data=data)


class TypesTest(tf.test.TestCase):
    def test_basic_lookup_eq(self):
        l1 = basic_lookup()
        l2 = basic_lookup()

        self.assertEqual(l1, l2)

    def test_basic_lookup_rank_not_eq(self):
        l1 = basic_lookup()
        l2 = basic_lookup(rank=2)

        self.assertNotEqual(l1, l2)

    def test_basic_lookup_distance_not_eq(self):
        l1 = basic_lookup()
        l2 = basic_lookup(distance=0.3)

        self.assertNotEqual(l1, l2)

    def test_label_lookup_eq(self):
        l1 = label_lookup()
        l2 = label_lookup()

        self.assertEqual(l1, l2)

    def test_label_lookup_not_eq(self):
        l1 = label_lookup()
        l2 = label_lookup(label=20)

        self.assertNotEqual(l1, l2)

    def test_optional_label_lookup_not_eq(self):
        l1 = label_lookup()
        l2 = label_lookup(label=None)

        self.assertNotEqual(l1, l2)

    def test_embedding_lookup_eq(self):
        l1 = embedding_lookup(embedding=np.ones((1, 512)))
        l2 = embedding_lookup(embedding=np.ones((1, 512)))

        self.assertEqual(l1, l2)

    def test_embedding_lookup_not_eq(self):
        l1 = embedding_lookup(embedding=np.ones((1, 512)))
        l2 = embedding_lookup(embedding=np.ones((1, 512)) + 1)

        self.assertNotEqual(l1, l2)

    def test_optional_embedding_lookup_not_eq(self):
        l1 = embedding_lookup(embedding=np.ones((1, 512)))
        l2 = embedding_lookup(embedding=None)

        self.assertNotEqual(l1, l2)

    def test_data_lookup_eq(self):
        l1 = data_lookup(data=tf.constant("foo"))
        l2 = data_lookup(data=tf.constant("foo"))

        self.assertEqual(l1, l2)

    def test_data_lookup_not_eq(self):
        l1 = data_lookup(data=tf.constant("foo"))
        l2 = data_lookup(data=tf.constant("bar"))

        self.assertNotEqual(l1, l2)

    def test_optional_data_lookup_not_eq(self):
        l1 = data_lookup(data=tf.constant("foo"))
        l2 = data_lookup(data=None)

        self.assertNotEqual(l1, l2)

    def test_lookup_does_not_match_wrong_class(self):
        class WrongClass:
            def __init__(self):
                self.rank = 1
                self.distance = 0.1
                self.label = None
                self.embedding = None
                self.data = None

        l1 = basic_lookup()
        l2 = WrongClass()

        self.assertNotEqual(l1, l2)


if __name__ == "__main__":
    tf.test.main()
