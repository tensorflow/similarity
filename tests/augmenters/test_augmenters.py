import pytest
import tensorflow as tf

from tensorflow_similarity.augmenters import BarlowAugmenter, SimCLRAugmenter


def create_imgs(width=32, height=32, channels=3, num=5):
    return tf.random.uniform([num, width, height, channels], 0, 1)


def test_barlow():
    imgs = create_imgs()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    NUM = 5

    aug = BarlowAugmenter(WIDTH, HEIGHT)
    augmented = aug.augment(imgs)

    assert (tf.shape(augmented) == tf.constant([2, NUM, WIDTH, HEIGHT, CHANNELS])).numpy().all()


def test_simclr():
    imgs = create_imgs()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    NUM = 5

    aug = SimCLRAugmenter(HEIGHT, WIDTH)
    augmented = aug.augment(imgs, tf.constant([0]), 2, True)

    assert (tf.shape(augmented) == tf.constant([2, NUM, WIDTH, HEIGHT, CHANNELS])).numpy().all()
