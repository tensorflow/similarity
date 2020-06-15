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

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from tensorflow_similarity.api.engine.augmentation import Augmentation
from tensorflow_similarity.utils.config_utils import register_custom_object


def sometimes(aug): return iaa.Sometimes(0.125, aug)


HEAVY_SEQUENCE = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.1),  # horizontally flip 50% of all images
        iaa.Flipud(0.1),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(
            iaa.CropAndPad(
                percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2)
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.2, 0.2)
                },  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[
                    0,
                    1
                ],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(
                    0,
                    255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.
                # use any of scikit-image's warping modes (see 2nd image from
                # the top for examples)
                ALL
            )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf(
            (0, 5),
            [
                sometimes(
                    iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
                ),  # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur(
                        (0,
                         3.0)),  # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(
                        k=(2, 7)
                    ),  # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(
                        k=(3, 11)
                    ),  # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75,
                                                       1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(
                    iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255),
                    per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5
                                ),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15),
                                      size_percent=(0.02, 0.05),
                                      per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                iaa.Add(
                    (-10, 10), per_channel=0.5
                ),  # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation(
                    (-20, 20)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0)))
                ]),
                iaa.ContrastNormalization(
                    (0.5, 2.0),
                    per_channel=0.5),  # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                # sigma=0.25)), # move pixels locally around (with random
                # strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))
                          ),  # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True)
    ],
    random_order=True)

LIGHT_SEQUENCE = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={
                "x": (0.8, 1.2),
                "y": (0.8, 1.2)
            },
            translate_percent={
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2)
            },
            rotate=(-25, 25),
            shear=(-8, 8))
    ],
    random_order=True)  # apply augmenters in random order


class ImageAugmentation(Augmentation):
    def __init__(self, heavy=False):
        self.heavy = heavy
        if heavy:
            self.sequence = HEAVY_SEQUENCE
        else:
            self.sequence = LIGHT_SEQUENCE

    def augment(self, x):
        x = x["example"]

        aug = np.array(self.sequence.augment_images([x]))
        return {
            "example": aug
        }

    def get_config(self):
        return {'heavy': self.heavy}


register_custom_object('ImageAugmentation', ImageAugmentation)


class PreprocessTo2D(object):
    def __init__(self):
        pass

    def __call__(self, img):
        o = []
        if len(np.shape(img)) > 2:
            for row in img:
                o_row = []
                for item in row:
                    o_row.append(item[0])
                o.append(o_row)
        else:
            return img

        return np.array(o)

    def get_config(self):
        return {"class_name": self.__class__.__name__, "config": {}}


register_custom_object('PreprocessTo2D', PreprocessTo2D)

if __name__ == '__main__':
    aug = ImageAugmentation()

    import cv2
    img = cv2.imread("test.png")
    print(img)

    for i in range(10):
        cv2.imwrite("test_%d.png" % i, aug(img))

    print(aug(img))
