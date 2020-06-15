# Copyright 2020 Google LLC
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

from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from tensorflow_similarity.preprocessing import images
from tensorflow_similarity.preprocessing.images import Resize48x48
from tensorflow_similarity.utils.config_utils import register_custom_object

# Performance data is available:
# https://imgaug.readthedocs.io/en/latest/source/performance.html


class ImageAugmentation(object):
    def __init__(self):
        self.blur = iaa.GaussianBlur((.01, 1.0))  # 5000/13000
        self.salt = iaa.CoarseSalt(.1, size_percent=0.02)

        self.darken = iaa.GammaContrast(gamma=(1.01, 1.125))  # 450/720
        self.brighten = iaa.GammaContrast(gamma=(0.8, 0.99))  # 450/720
        self.illumination = iaa.OneOf([self.darken, self.brighten])  # 800/3000

        self.light_occlusion_prob = 0.05
        self.light_occlusion_size = 0.03
        self.light_occlusion = iaa.SomeOf((1, 2), [
            iaa.CoarsePepper(self.light_occlusion_prob,
                             size_percent=self.light_occlusion_size),
            iaa.CoarseSalt(self.light_occlusion_prob,
                           size_percent=self.light_occlusion_size),
            iaa.CoarseSaltAndPepper(
                self.light_occlusion_prob, size_percent=self.light_occlusion_size),
            iaa.CoarseDropout(self.light_occlusion_prob,
                              size_percent=self.light_occlusion_size),
            # iaa.Superpixels(p_replace=0.3, n_segments=32) # 10.4 / 10.9
        ], random_order=True)

        self.heavy_occlusion_prob_range = (.1, .2)
        self.heavy_occlusion_size_range = (.15, .25)
        self.heavy_occlusion = iaa.Sequential(
            [
                # iaa.Superpixels(p_replace=0.1, n_segments=32), # 10.4 / 10.9
                iaa.OneOf([
                    iaa.CoarsePepper(self.heavy_occlusion_prob_range,  # ~1500
                                     size_percent=self.heavy_occlusion_prob_range),
                    iaa.CoarseSalt(self.heavy_occlusion_prob_range,  # ~1500
                                   size_percent=self.heavy_occlusion_prob_range),
                    iaa.CoarseSaltAndPepper(
                        self.heavy_occlusion_prob_range, size_percent=self.heavy_occlusion_prob_range),  # ~1500
                    iaa.CoarseDropout(
                        self.heavy_occlusion_prob_range, size_percent=self.heavy_occlusion_prob_range),  # ~2700
                ])
            ])

        self.occlusion = iaa.OneOf(
            [self.light_occlusion, self.heavy_occlusion])

        self.channel_shuffle = iaa.ChannelShuffle(p=.10)

        self.affine_augmentations = iaa.SomeOf((1, 4), [
            iaa.Fliplr(1),  # 1526 / 6055.7
            iaa.Flipud(1),  # 1503 / 6070
            iaa.Affine(shear=(-16, 16)),  # 2000/5000
            iaa.Affine(translate_percent={
                       'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}),  # 2000/5000
            iaa.Affine(rotate=(-25, 25)),  # 2000/5000
            iaa.Affine(scale={'x': (0.8, 1.5), 'y': (0.8, 1.5)}),  # 2000/5000

        ], random_order=True)

        self.color_augmentations = iaa.Sometimes(
            .35,
            iaa.OneOf([
                self.channel_shuffle,
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.Add((-40, 40), per_channel=.5)
            ]))

        self.augmentation_sequence = iaa.Sequential([
            self.color_augmentations,
            self.heavy_occlusion,
            self.affine_augmentations])

    def __call__(self, x):
        if x is None:
            return None

        if x.dtype != np.uint8:
            x = x.astype(np.uint8)

        inp = [x]

        return self.augmentation_sequence.augment_images(inp)[0]

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
            }
        }


register_custom_object("ImageAugmentation", ImageAugmentation)
