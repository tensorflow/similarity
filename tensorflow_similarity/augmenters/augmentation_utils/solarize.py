import tensorflow as tf

from tensorflow_similarity.augmenters.augmentation_utils.random_apply import (
    random_apply,
)
from tensorflow_similarity.types import Tensor


def random_solarize(
    image: Tensor,
    p: float = 0.2,
    pixel_min=0,
    pixel_max=255,
    thresh: int = 10,
) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        return tf.where(image < thresh, image, pixel_max - image)

    return random_apply(_transform, p=p, x=image)
