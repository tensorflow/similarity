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


@tf.function
def augment_barlow(image: tf.Tensor, height: int, width: int):
    image = random_crop_with_resize(image, height, width)
    image = random_random_flip_left_right(image)
    image = random_color_jitter(image,
                                impl="barlow")
    image = random_blur(image, height, width, 0.2)
    image = random_solarize(image)
    image = tf.clip_by_value(image, 0, 1)

    return image


class BarlowAugmenter(Augmenter):
    def __init__(self,
                 width: int,
                 height: int,
                 num_cpu: Optional[int] = os.cpu_count(),
         ):
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
            inputs = tf.cast(inputs, dtype="float32") / 255.0
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
