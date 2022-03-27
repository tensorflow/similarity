import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow_similarity.augmenters.augmenter import Augmenter
from typing import Callable, List, Optional, Tuple, Any
import os


from tensorflow_similarity.augmenters.augmentation_utils.cropping import random_crop_with_resize
from tensorflow_similarity.augmenters.augmentation_utils.flip import random_random_flip_left_right
from tensorflow_similarity.augmenters.augmentation_utils.color_jitter import random_color_jitter
from tensorflow_similarity.augmenters.augmentation_utils.blur import random_blur
from tensorflow_similarity.augmenters.augmentation_utils.solarize import random_solarize
from tensorflow_similarity.augmenters.augmentation_utils.random_apply import random_apply


@tf.function
def augment_barlow(
    image: tf.Tensor, 
    height: int, 
    width: int,
    brighness_multiplier=0.8,
    contrast_multiplier=0.6,
    saturation_multiplier=0.6,
    hue_multiplier=0.2,
    jitter_probability=0.8,
    greyscale_probability=0.2,
    blur_probability=0.2,
    blur_min_sigma=0,
    blur_max_sigma=1,
    solarize_probability=0.2,
    solarize_thresh=10
):
    image = tf.cast(image, dtype="float32") / 255.0
    image = random_crop_with_resize(image, height, width)
    image = random_random_flip_left_right(image)
    image = random_color_jitter(
      image,
      p_jitter=jitter_probability,
      p_grey= greyscale_probability,
      strength=1.0,
      brightness_multiplier=brightness_multiplier, 
      contrast_multiplier=contrast_multiplier, 
      saturation_multiplier=saturation_multiplier, 
      hue_multiplier=hue_multiplier,
      impl="additive"
    )
    image = random_blur(
      image=image, 
      height=height, 
      width=width, 
      p=blur_probability,
      min_sigma=blur_min_sigma,
      max_sigma=blur_max_sigma
    )
    image = random_solarize(
        image, thresh=solarize_thresh, p=solarize_probability
    )
    image = tf.clip_by_value(image, 0, 1)

    return image


class BarlowAugmenter(Augmenter):
    def __init__(self,
                 width: int,
                 height: int,
                 num_cpu: Optional[int] = os.cpu_count(),
         ):
        print("HI")
        super(Augmenter, self).__init__()
        self.num_cpu = num_cpu
        self.width = width
        self.height = height
    
    @tf.function
    def augment(
        self,
        x: Any,
        y: Any = tf.constant([0]),
        num_augmentations_per_example: int = 2,
        is_warmup: bool = True,
    ) -> List[Any]:

        with tf.device("/cpu:0"):
            inputs = tf.stack(x)
#             inputs = tf.cast(inputs, dtype="float32") / 255.0
            views = []

            for _ in range(num_augmentations_per_example):

                view = tf.map_fn(lambda img: augment_barlow(img, self.width, self.height),
                                 inputs,
                                 parallel_iterations=self.num_cpu)
                views.append(view)
        return views
    

    def __call__(
        self, x: Any, y: Any = tf.constant([0]), num_augmentations_per_example: int = 2, is_warmup: bool = True,
    ) -> List[Any]:
        return list(self.augment(x))
