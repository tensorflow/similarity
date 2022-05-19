import tensorflow as tf

from tensorflow_similarity.augmenters.augmentation_utils.random_apply import (
    random_apply,
)


def random_random_flip_left_right(
    image: tf.Tensor, p: float = 0.5
) -> tf.Tensor:
    def _transform(image: tf.Tensor) -> tf.Tensor:
        return tf.image.random_flip_left_right(image)

    return random_apply(_transform, p=p, x=image)


def random_random_flip_top_bottom(
    image: tf.Tensor, p: float = 0.5
) -> tf.Tensor:
    def _transform(image: tf.Tensor) -> tf.Tensor:
        return tf.image.random_flip_up_down(image)

    return random_apply(_transform, p=p, x=image)
