import pytest
from tensorflow_similarity.augmenters.augmentation_utils import cropping
import tensorflow as tf 

def create_img(width=32, height=32, channels=3):
  return tf.random.uniform(
    [width, height, channels], 0, 1)
  
def test_center_cropping():
    img = create_img()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3

    center_cropped = cropping.center_crop(img, HEIGHT, WIDTH, 0.5)

    assert (tf.shape(center_cropped) == tf.shape(img)).numpy().all()
    
  
def test_random_cropping():
    # Random Crop
    img = create_img()
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3

    random_cropping_always = cropping.random_crop_with_resize(
        img, HEIGHT, WIDTH, 1
    )

    random_cropping_never = cropping.random_crop_with_resize(
        img, HEIGHT, WIDTH, 0
    )

    # check shapes
    assert (tf.shape(random_cropping_always) == tf.shape(img)).numpy().all()
    assert (tf.shape(random_cropping_never) == tf.shape(img)).numpy().all()

    # check if crop works
    assert not (random_cropping_always == img).numpy().all()
    assert (random_cropping_never == img).numpy().all()
