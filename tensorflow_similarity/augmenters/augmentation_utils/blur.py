from typing import List

import tensorflow as tf

from tensorflow_similarity.augmenters.augmentation_utils.random_apply import (
    random_apply,
)
from tensorflow_similarity.types import FloatTensor, Tensor


def random_blur(
    image: Tensor,
    height: int,
    width: int,
    p: float = 1.0,
    min_sigma: float = 0.1,
    max_sigma: float = 2.0,
    kernel_size_divider: float = 10,
) -> Tensor:
    """Randomly blur an image.

    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      p: probability of applying this transformation.

    Returns:
      A preprocessed image `Tensor`.
    """
    del width

    def _transform(image: Tensor) -> Tensor:
        sigma = tf.random.uniform([], min_sigma, max_sigma, dtype=tf.float32)
        return gaussian_blur(
            image,
            kernel_size=height // kernel_size_divider,
            sigma=sigma,
            padding="SAME",
        )

    return random_apply(_transform, p=p, x=image)


def batch_random_blur(
    images_list: List[Tensor],
    height: int,
    width: int,
    blur_probability: float = 0.5,
) -> List[Tensor]:
    """Apply efficient batch data transformations.

    Args:
      images_list: a list of image tensors.
      height: the height of image.
      width: the width of image.
      blur_probability: the probaility to apply the blur operator.

    Returns:
      Preprocessed feature list.
    """

    def generate_selector(p: float, bsz: int) -> FloatTensor:
        shape = [bsz, 1]
        cond = tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p)
        selector: FloatTensor = tf.cast(cond, tf.float32)
        return selector

    new_images_list = []
    for images in images_list:
        images_new = random_blur(images, height, width, p=1.0)
        selector = generate_selector(blur_probability, tf.shape(images)[0])
        images = images_new * selector + images * (1 - selector)
        images = tf.clip_by_value(images, 0.0, 1.0)
        new_images_list.append(images)

    return new_images_list


def gaussian_blur(
    image: Tensor,
    kernel_size: Tensor,
    sigma: float,
    padding: str = "SAME",
) -> Tensor:
    """Blurs the given image with separable convolution.


    Args:
      image: Tensor of shape [height, width, channels] and dtype float to blur.
      kernel_size: Integer Tensor for the size of the blur kernel. This is
      should be an odd number. If it is an even number, the actual kernel
      size will besize + 1.
      sigma: Sigma value for gaussian operator.
      padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

    Returns:
      A Tensor representing the blurred image.
    """
    radius = tf.cast(kernel_size / 2, dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions,
        # which we can fake with an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred
