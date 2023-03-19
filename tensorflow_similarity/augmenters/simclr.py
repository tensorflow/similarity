# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data preprocessing and augmentation."""
from __future__ import annotations

import os

import tensorflow as tf

from tensorflow_similarity.augmenters import Augmenter
from tensorflow_similarity.types import Tensor

from .augmentation_utils.color_jitter import random_color_jitter
from .augmentation_utils.cropping import center_crop, random_crop_with_resize


def simclr_training_augmentation(
    image: Tensor,
    height: int,
    width: int,
    color_distort: bool = True,
    jitter_stength: float = 1.0,
    crop: bool = True,
    flip: bool = True,
    impl: str = "multiplicative",
) -> Tensor:
    """SimCLR Preprocesses the given image for training.

    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      color_distort: Whether to apply the color distortion.
      crop: Whether to crop the image.
      flip: Whether or not to flip left and right of an image.
      impl: 'additive' or 'multiplicative'.  Whether to use simclrv1 or simclrv2's
          version of random brightness.

    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter(image, strength=jitter_stength, impl=impl)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def simclr_eval_augmentation(
    image: Tensor,
    height: int,
    width: int,
    crop: bool = True,
    crop_proportion: float = 0.875,
) -> Tensor:
    """Preprocesses the given image for evaluation.

    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      crop: Whether or not to (center) crop the test images.

    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = center_crop(image, height, width, crop_proportion=crop_proportion)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


class SimCLRAugmenter(Augmenter):
    """SimCLR augmentation pipeline as defined in
    [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)

    code adapted from [orignal github](https://github.com/google-research/simclr/tree/master/tf2)
    """

    def __init__(
        self,
        height: int,
        width: int,
        is_training: bool = True,
        color_distort: bool = True,
        jitter_stength: float = 1.0,
        crop: bool = True,
        eval_crop_proportion: float = 0.875,  # imagenet standard
        flip: bool = True,
        version: str = "v2",
        num_cpu: int | None = os.cpu_count(),
    ):

        self.width = width
        self.height = height
        self.is_training = is_training
        self.color_distort = color_distort
        self.jitter_stength = jitter_stength
        self.crop = crop
        self.eval_crop_proportion = eval_crop_proportion
        self.flip = flip
        self.version = version
        self.num_cpu = num_cpu

        if version == "v2":
            self.impl = "multiplicative"
        else:
            self.impl = "additive"

        if self.is_training:
            self.augment_img = self._train_augment_img
        else:
            self.augment_img = self._eval_augment_img

    @tf.function
    def augment(self, x: Tensor, y: Tensor, num_views: int, is_warmup: bool) -> list[Tensor]:

        with tf.device("/cpu:0"):
            inputs = tf.stack(x)
            inputs = tf.cast(inputs, dtype="float32") / 255.0
            views = []

            for _ in range(num_views):
                # multi-cor augementations
                view = tf.map_fn(self.augment_img, inputs, parallel_iterations=self.num_cpu)
                views.append(view)
        return views

    def _train_augment_img(self, img: Tensor) -> Tensor:
        return simclr_training_augmentation(
            img,
            self.height,
            self.width,
            self.color_distort,
            self.jitter_stength,
            self.crop,
            self.flip,
            self.impl,
        )

    def _eval_augment_img(self, img: Tensor) -> Tensor:
        return simclr_eval_augmentation(img, self.height, self.width, self.crop, self.eval_crop_proportion)
