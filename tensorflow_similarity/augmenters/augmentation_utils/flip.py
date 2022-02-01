import os
import functools
from typing import Callable, List, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow_similarity.types import FloatTensor
from tensorflow_similarity.augmenters.augmentation_utils.random_apply import random_apply

def random_random_flip_left_right(
 image: Tensor, p: float = 0.5
) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        return tf.image.random_flip_left_right(image)

    return random_apply(_transform, p=p, x=image)

def random_random_flip_top_bottom(
    image: Tensor, p: float = 0.5
) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        return tf.image.random_flip_up_down(image)

    return random_apply(_transform, p=p, x=image)
