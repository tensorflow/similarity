import pytest
from tensorflow_similarity.augmenters.augmentation_utils import solarize
import tensorflow as tf 

def create_img(width=32, height=32, channels=3):
  return tf.random.uniform(
    [width, height, channels], 0, 1)
  
def test_solarization():
  img = create_img()

  random_solarize_always = solarize.random_solarize(img, p=1, thresh=0, pixel_min=0, pixel_max=1)
  random_solarize_never = solarize.random_solarize(img, p=0, thresh=0, pixel_min=0, pixel_max=1)

  # check shapes
  assert (tf.shape(random_solarize_always) == tf.shape(img)).numpy().all()
  assert (tf.shape(random_solarize_never) == tf.shape(img)).numpy().all()

  # check if flip works
  assert not (random_solarize_always == img).numpy().all()
  assert (random_solarize_never == img).numpy().all()
