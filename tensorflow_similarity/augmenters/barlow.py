import tensorflow as tf
from tensorflow import keras
import numpy as np
from augmenter import Augmenter
from typing import Callable, List, Optional, Tuple, Any
import os


from augmentation_utils.cropping import random_crop_with_resize
from augmentation_utils.flip import random_random_flip_left_right
from augmentation_utils.color_jitter import random_color_jitter
from augmentation_utils.blur import random_blur
from augmentation_utils.solarize import random_solarize


def augment_barlow(image: tf.Tensor, height: int, width: int):
    image = image / 255.0
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
                 num_cpu: Optional[int] = os.cpu_count(),
                 seed: Optional[int] = 128):
        super(Augmenter, self).__init__()
        self.num_cpu = num_cpu
        self.seed = seed

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
                view = tf.map_fn(augment_barlow,
                                 x,
                                 parallel_iterations=self.num_cpu)
                views.append(view)
        return views
    

    def __call__(
        self, x: Any, y: Any = tf.constant([0]), num_augmentations_per_example: int = 2, is_warmup: bool = True,
    ) -> List[Any]:
        return self.augment(x)

