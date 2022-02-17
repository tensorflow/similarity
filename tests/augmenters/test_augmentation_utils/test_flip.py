import pytest
from tensorflow_similarity.augmenters.augmentation_utils import flip
import tensorflow as tf 

def create_img(width=32, height=32, channels=3):
  return tf.random.uniform(
    [width, height, channels], 0, 1)
  
def test_flip_left_right():
  img = create_img()

  random_flip_always = flip.random_random_flip_left_right(img, 1)
  random_flip_never = flip.random_random_flip_left_right(img, 0)

  # check shapes
  assert (tf.shape(random_flip_always) == tf.shape(img)).numpy().all()
  assert (tf.shape(random_flip_never) == tf.shape(img)).numpy().all()

  # check if flip works
  assert not (random_flip_always == img).numpy().all()
  assert (random_flip_never == img).numpy().all()

def test_flip_top_bottom():
  img = create_img()

  random_flip_always = flip.random_random_flip_top_bottom(img, 1)
  random_flip_never = flip.random_random_flip_top_bottom(img, 0)

  # check shapes
  assert (tf.shape(random_flip_always) == tf.shape(img)).numpy().all()
  assert (tf.shape(random_flip_never) == tf.shape(img)).numpy().all()

  # check if flip works
  assert not (random_flip_always == img).numpy().all()
  assert (random_flip_never == img).numpy().all()
