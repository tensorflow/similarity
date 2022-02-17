
import pytest
from tensorflow_similarity.augmenters.augmentation_utils import color_jitter
import tensorflow as tf 

def create_img(width=32, height=32, channels=3):
  return tf.random.uniform(
    [width, height, channels], 0, 1)
  
def test_random_color_jitter_simclrv2():
    # Random Color Jitter
    img = create_img()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3

    random_jitter_always = color_jitter.random_color_jitter(
        img, 1, 1, 1, impl="simclrv2"
    )

    random_jitter_never = color_jitter.random_color_jitter(
        img, 0, impl="simclrv2"
    )

    # check shapes
    assert (tf.shape(random_jitter_always) == tf.shape(img)).numpy().all()
    assert (tf.shape(random_jitter_never) == tf.shape(img)).numpy().all()

    # check if blur works
    assert not (random_jitter_always == img).numpy().all()
    assert (random_jitter_never == img).numpy().all()

def test_random_color_jitter_other():
    # Random Color Jitter
    img = create_img()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3

    random_jitter_always = color_jitter.random_color_jitter(
        img, 1, 1, 1, impl="barlow" # won't make a difference between barlow/v1
    )

    random_jitter_never = color_jitter.random_color_jitter(
        img, 0, impl="simclrv1" # won't make a difference between barlow/v1
    )

    # check shapes
    assert (tf.shape(random_jitter_always) == tf.shape(img)).numpy().all()
    assert (tf.shape(random_jitter_never) == tf.shape(img)).numpy().all()

    # check if color jitter works
    assert not (random_jitter_always == img).numpy().all()
    assert (random_jitter_never == img).numpy().all()
