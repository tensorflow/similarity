import os
import functools
from typing import Callable, List, Optional, Tuple

import tensorflow as tf
from tensorflow import Tensor
from tensorflow_similarity.types import FloatTensor

from tensorflow_similarity.augmenters.augmentation_utils.random_apply import random_apply

def color_jitter(
    image: Tensor,
    strength: float,
    random_order: bool = True,
    impl: str = "simclrv2",
) -> Tensor:
    """Distorts the color of the image.

    Args:
      image: The input image tensor.
      strength: the floating number for the strength of the color augmentation.
      random_order: A bool, specifying whether to randomize the jittering order.
      impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
          version of random brightness.

    Returns:
      The distorted image tensor.
    """
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(
            image, brightness, contrast, saturation, hue, impl=impl
        )
    else:
        return color_jitter_nonrand(
            image, brightness, contrast, saturation, hue, impl=impl
        )

def color_jitter_nonrand(
    image: Tensor,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    hue: float = 0,
    impl: str = "simclrv2",
) -> Tensor:
    """Distorts the color of the image (jittering order is fixed).

    Args:
      image: The input image tensor.
      brightness: A float, specifying the brightness for color jitter.
      contrast: A float, specifying the contrast for color jitter.
      saturation: A float, specifying the saturation for color jitter.
      hue: A float, specifying the hue for color jitter.
      impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
          version of random brightness.

    Returns:
      The distorted image tensor.
    """
    with tf.name_scope("distort_color"):

        def apply_transform(
            i: int,
            x: Tensor,
            brightness: float,
            contrast: float,
            saturation: float,
            hue: float,
        ) -> Tensor:
            """Apply the i-th transformation."""
            if brightness != 0 and i == 0:
                x = random_brightness(x, max_delta=brightness, impl=impl)
            elif contrast != 0 and i == 1:
                x = tf.image.random_contrast(
                    x, lower=1 - contrast, upper=1 + contrast
                )
            elif saturation != 0 and i == 2:
                x = tf.image.random_saturation(
                    x, lower=1 - saturation, upper=1 + saturation
                )
            elif hue != 0:
                x = tf.image.random_hue(x, max_delta=hue)
            return x

        for i in range(4):
            image = apply_transform(
                i, image, brightness, contrast, saturation, hue
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image

def color_jitter_rand(
    image: Tensor,
    brightness: float = 0,
    contrast: float = 0,
    saturation: float = 0,
    hue: float = 0,
    impl: str = "simclrv2",
) -> Tensor:
    """Distorts the color of the image (jittering order is random).

    Args:
      image: The input image tensor.
      brightness: A float, specifying the brightness for color jitter.
      contrast: A float, specifying the contrast for color jitter.
      saturation: A float, specifying the saturation for color jitter.
      hue: A float, specifying the hue for color jitter.
      impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
          version of random brightness.

    Returns:
      The distorted image tensor.
    """
    with tf.name_scope("distort_color"):

        def apply_transform(i, x):
            """Apply the i-th transformation."""

            def brightness_foo():
                if brightness == 0:
                    return x
                else:
                    return random_brightness(x, max_delta=brightness, impl=impl)

            def contrast_foo():
                if contrast == 0:
                    return x
                else:
                    return tf.image.random_contrast(
                        x, lower=1 - contrast, upper=1 + contrast
                    )

            def saturation_foo():
                if saturation == 0:
                    return x
                else:
                    return tf.image.random_saturation(
                        x, lower=1 - saturation, upper=1 + saturation
                    )

            def hue_foo():
                if hue == 0:
                    return x
                else:
                    return tf.image.random_hue(x, max_delta=hue)

            x = tf.cond(
                tf.less(i, 2),
                lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo),
            )
            return x

        perm = tf.random.shuffle(tf.range(4))
        for i in range(4):
            image = apply_transform(perm[i], image)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image

def to_grayscale(image: Tensor, keep_channels: bool = True) -> Tensor:
    image = tf.image.rgb_to_grayscale(image)
    if keep_channels:
        image = tf.tile(image, [1, 1, 3])
    return image

def random_brightness(
    image: Tensor, max_delta: float, impl: str = "simclrv2"
) -> Tensor:
    """A multiplicative vs additive change of brightness."""
    if impl == "simclrv2":
        factor = tf.random.uniform(
            [], tf.maximum(1.0 - max_delta, 0), 1.0 + max_delta
        )
        image = image * factor
    elif impl == "simclrv1" or impl == "barlow":
        image = tf.image.random_brightness(image, max_delta=max_delta)
    else:
        raise ValueError("Unknown impl {} for random brightness.".format(impl))
    return image

def random_color_jitter(
    image: Tensor, p_execute = 1.0, p_jitter: float = 0.8, p_grey: float = 0.2, strength: float = 1.0, impl: str = "simclrv2"
) -> Tensor:
    def _transform(image: Tensor) -> Tensor:
        color_jitter_t = functools.partial(
            color_jitter, strength=strength, impl=impl
        )
        image = random_apply(color_jitter_t, p=p_jitter, x=image)
        return random_apply(to_grayscale, p=p_grey, x=image)

    return random_apply(_transform, p=p_execute, x=image)
