import pytest 
from tensorflow_similarity.augmenters.augmentation_utils import blur 
import tensorflow as tf

def create_img(width=32, height=32, channels=3):
  return tf.random.uniform(
    [width, height, channels], 0, 1)

def test_random_blur():
  # Random Blur
  img = create_img()
  WIDTH = 32
  HEIGHT = 32
  CHANNELS = 3

  random_blurred_always = blur.random_blur(
      img, 32, 32, 1.0
  )

  random_blurred_never = blur.random_blur(
      img, 32, 32, 0
  )

  # check shapes
  assert (tf.shape(random_blurred_always) == tf.shape(img)).numpy().all()
  assert (tf.shape(random_blurred_never) == tf.shape(img)).numpy().all()

  # check if blur works
  assert not (random_blurred_always == img).numpy().all()
  assert (random_blurred_never == img).numpy().all()

def test_batch_random_blur():
    img = create_img()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3

    batched_img = [img]

    random_batched_blurred_always = blur.batch_random_blur(
        batched_img, 32, 32, 1.0
    )

    random_batched_blurred_never = blur.batch_random_blur(
        batched_img, 32, 32, 0
    )

    # check shapes
    assert (tf.shape(random_batched_blurred_always)
            == tf.shape(batched_img)).numpy().all()
    assert (tf.shape(random_batched_blurred_never)  
            == tf.shape(batched_img)).numpy().all()

    # check if blur works
    equality_always = tf.reshape(
        tf.equal(random_batched_blurred_always, batched_img), [-1])
    equality_never = tf.reshape(
        tf.equal(random_batched_blurred_never, batched_img), [-1])

    assert not equality_always.numpy().all()
    assert equality_never.numpy().all()
    
