import tensorflow as tf
from tensorflow import Tensor 
from tensorflow_similarity.augmenters.augmentation_utils.random_apply import random_apply

def random_solarize(
    image: Tensor, p: float = 0.2, thresh: int = 10
) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        return tf.where(image < 10/255, image, 255/255 - image)

    return random_apply(_transform, p=p, x=image)

