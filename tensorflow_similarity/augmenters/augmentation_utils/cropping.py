# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import tensorflow as tf

from tensorflow_similarity.augmenters.augmentation_utils.random_apply import (
    random_apply,
)
from tensorflow_similarity.types import Tensor


def _compute_crop_shape(
    image_height: int,
    image_width: int,
    aspect_ratio: float,
    crop_proportion: float,
) -> tuple[int, int]:
    """Compute aspect ratio-preserving shape for central crop.

    The resulting shape retains `crop_proportion` along one side and a
    proportion less than or equal to `crop_proportion` along the other side.

    Args:
      image_height: Height of image to be cropped.
      image_width: Width of image to be cropped.
      aspect_ratio: Desired aspect ratio (width / height) of output.
      crop_proportion: Proportion of image to retain along the less-cropped
      side.

    Returns:
      crop_height: Height of image after cropping.
      crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(
            tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
            tf.int32,
        )
        crop_width = tf.cast(tf.math.rint(crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(tf.math.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(
            tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
            tf.int32,
        )
        return crop_height, crop_width

    crop_height, crop_width = tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio,
    )

    return crop_height, crop_width


def center_crop(image: Tensor, height: int, width: int, crop_proportion: float) -> Tensor:
    """Crops to center of image and rescales to desired size.

    Args:
      image: Image Tensor to crop.
      height: Height of image to be cropped.
      width: Width of image to be cropped.
      crop_proportion: Proportion of image to retain along the less-cropped side.

    Returns:
      A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)

    image = tf.image.resize([image], [height, width], method=tf.image.ResizeMethod.BICUBIC)[0]

    return image


def distorted_bounding_box_crop(
    image: Tensor,
    bbox: Tensor,
    min_object_covered: float = 0.1,
    aspect_ratio_range: tuple[float, float] = (0.75, 1.33),
    area_range: tuple[float, float] = (0.05, 1.0),
    max_attempts: int = 100,
    scope: bool | None = None,
) -> Tensor:
    """Generates cropped_image using one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image: `Tensor` of image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      cropped image `Tensor`.
    """
    with tf.name_scope(scope or "distorted_bounding_box_crop"):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True,
        )
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)

        return image


def crop_and_resize(
    image: Tensor,
    height: int,
    width: int,
    area_range: tuple[float, float] = (0.08, 1.0),
) -> Tensor:
    """Make a random crop and resize it to height `height` and width `width`.

    Args:
      image: Tensor representing the image.
      height: Desired image height.
      width: Desired image width.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.

    Returns:
      A `height` x `width` x channels Tensor holding a random crop of `image`.
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
        image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3.0 / 4 * aspect_ratio, 4.0 / 3.0 * aspect_ratio),
        area_range=area_range,
        max_attempts=100,
        scope=None,
    )
    img = tf.image.resize([image], [height, width], method=tf.image.ResizeMethod.BICUBIC)[0]
    return img


def crop_random_resized_crop(
    image: Tensor,
    height: int,
    width: int,
    min_cropsize_multiplier: float = 0.75,
    max_cropsize_multiplier: float = 1,
):
    """
    Crops and resizes image by getting a random square of size ranging from 75% of the width to 100% of width.
    This cropped portion is then resized to the original height and width, which may cause stretching and
    distortion for rectangular images.
    """

    rand_width = tf.random.uniform(
        shape=[],
        minval=int(min_cropsize_multiplier * width),
        maxval=int(max_cropsize_multiplier * width),
        dtype=tf.int32,
    )

    crop = tf.image.random_crop(image, (rand_width, rand_width, 3))
    crop_resize = tf.image.resize(crop, (height, width))
    return crop_resize


def random_resized_crop(
    # this utility function is supposed to match
    # torchvision.transforms.RandomResizedCrop
    # Only this will work with Barlow Twins and VicReg,
    # everything else will cause significant performance
    # losses
    image: Tensor,
    height: int,
    width: int,
    min_cropsize_multiplier: float = 0.75,
    max_cropsize_multiplier: float = 1,
    p: float = 1.0,
):
    def _transform(
        image: Tensor,
    ) -> Tensor:  # pylint: disable=missing-docstring
        image = crop_random_resized_crop(
            image,
            height,
            width,
            min_cropsize_multiplier,
            max_cropsize_multiplier,
        )
        return image

    return random_apply(_transform, p=p, x=image)


def random_crop_with_resize(image: Tensor, height: int, width: int, p: float = 1.0) -> Tensor:
    """Randomly crop and resize an image.

    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      p: Probability of applying this transformation.

    Returns:
      A preprocessed image `Tensor`.
    """

    def _transform(
        image: Tensor,
    ) -> Tensor:  # pylint: disable=missing-docstring
        image = crop_and_resize(image, height, width)
        return image

    return random_apply(_transform, p=p, x=image)
