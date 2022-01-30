import tensorflow as tf
from tensorflow import keras 
import tensorflow_addons as tfa
import numpy as np
from tensorflow_similarity.augmenters.augmenter import Augmenter 
from typing import Callable, List, Optional, Tuple, Any
import os

class Augmentation(keras.layers.Layer):
    def __init__(self):
        super(Augmentation, self).__init__()

    @tf.function
    def random_execute(self, prob: float) -> bool:
        return bool(tf.random.uniform([], minval=0, maxval=1) < prob)


class RandomToGrayscale(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:

        if self.random_execute(0.2):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x


class RandomColorJitter(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.8):
            x = tf.image.random_brightness(x, 0.8)
            x = tf.image.random_contrast(x, 0.4, 1.6)
            x = tf.image.random_saturation(x, 0.4, 1.6)
            x = tf.image.random_hue(x, 0.2)
        return x


class RandomFlip(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.5):
            x = tf.image.random_flip_left_right(x)
        return x


class RandomResizedCrop(Augmentation):
    def __init__(self):
        super(Augmentation, self).__init__()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x_dim = tf.shape(x)
        x_width = tf.cast(x_dim[0], tf.float32) 
        x_height = tf.cast(x_dim[1], tf.float32)

        print(x_width)
        rand_size_width = tf.random.uniform(
            shape=[],
            minval=int(0.75 * x_width),
            maxval=int(1 * x_width),
            dtype=tf.int32,
        )

        rand_size_height = tf.random.uniform(
            shape=[],
            minval=int(0.75 * x_height),
            maxval=int(1 * x_height),
            dtype=tf.int32,
        )

        crop = tf.image.random_crop(x, (x_width, x_height, 3))
        crop_resize = tf.image.resize(crop, (x_width, x_height))
        return crop_resize


class RandomSolarize(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.2):
            # flips abnormally low pixels to abnormally high pixels
            x = tf.where(x < 10, x, 255 - x)
        return x


class RandomBlur(Augmentation):
    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.random_execute(0.2):
            s = np.random.random()
            return tfa.image.gaussian_filter2d(image=x, sigma=s)
        return x


class BarlowTransformator(keras.Model):
    def __init__(self):
        super(BarlowTransformator, self).__init__()
        self.random_resized_crop = RandomResizedCrop()
        self.random_flip = RandomFlip()
        self.random_color_jitter = RandomColorJitter()
        self.random_blur = RandomBlur()
        self.random_to_grayscale = RandomToGrayscale()
        self.random_solarize = RandomSolarize()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.random_resized_crop(x)
        x = self.random_flip(x)
        x = self.random_color_jitter(x)
        x = self.random_blur(x)
        x = self.random_to_grayscale(x)
        x = self.random_solarize(x)

        x = tf.clip_by_value(x, 0, 1)
        return x

class BarlowAugmenter(Augmenter):
  def __init__(self, num_cpu: Optional[int] = os.cpu_count(), seed: Optional[int] = 128):
    super(Augmenter, self).__init__()
    self.num_cpu = num_cpu
    self.seed = seed

  def augment(
      self, x: Any, y: Any = tf.constant([0]), num_augmentations_per_example: int = 2, is_warmup: bool = True,
  ) -> List[Any]:

      with tf.device("/cpu:0"):
          inputs = tf.stack(x)
          inputs = tf.cast(inputs, dtype="float32") / 255.0
          views = []
          augmenter = BarlowTransformator()
          
          for _ in range(num_augmentations_per_example):
              view = (
                  tf.data.Dataset.from_tensor_slices(x)
                  .shuffle(1000, seed=self.seed)
                  .map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
                  .prefetch(tf.data.AUTOTUNE)
              )
              tolist = []
              for item in view:
                tolist.append(item)
              views.append(tolist)
      return views

  def __call__(
      self, x: Any, y: Any = tf.constant([0]), num_augmentations_per_example: int = 2, is_warmup: bool = True,
  ) -> List[Any]:
      return self.augment(x)

