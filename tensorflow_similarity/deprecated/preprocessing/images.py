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

from cairosvg import svg2png
import json
import logging
import os

import h5py
import numpy as np
import six
from PIL import Image
from six.moves import urllib
from tensorflow.keras.applications.mobilenet import \
    preprocess_input as mobilenet_batch_preprocessor
from tensorflow.keras.applications.mobilenet_v2 import \
    preprocess_input as mobilenet_v2_batch_preprocessor

from tensorflow_similarity.preprocessing.base import MoiraiPreprocessor
from tensorflow_similarity.utils.config_utils import register_custom_object
import tempfile


def normalize_image(x):
    x = np.divide(x, 255.0)
    #x = np.subtract(x, 0.5)
    #x = np.multiply(x, 2.0)
    return x


if six.PY2:
    VARLEN_STR = h5py.special_dtype(vlen=unicode)
else:
    VARLEN_STR = h5py.special_dtype(vlen=str)


def image_to_grayscale(img):
    dtype = img.dtype
    out = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return out.astype(np.uint8)


def image_file_to_bytes(filename, size=(32, 32)):
    filename_noext, extension = os.path.splitext(filename)

    if extension == ".svg":
        with open(filename, "rt") as f_img:
            svg = f_img.read()
        tmpfile = filename + ".png"
        svg2png(bytestring=svg, write_to=tmpfile)
        img = Image.open(tmpfile).resize(size)
        os.remove(tmpfile)
    else:
        img = Image.open(filename).resize(size)

    img = img.convert("RGBA")
    new_img = Image.new("RGB", img.size, (255, 255, 255))
    new_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel

    data = np.array(new_img)
    if len(data.shape) == 2:
        data = np.stack([data, data, data], axis=-1)

    # Grayscale + Alpha
    if len(data.shape) == 3 and data.shape[2] == 2:
        # Gray channel only - drop the alpha channel
        data = data[:, :, 0]
        #  Stack gray 3x to form a plausible RGB.
        data = np.stack([data, data, data], axis=-1)
    if len(data.shape) == 3 and data.shape[2] == 4:
        data = data[:, :, :3]

    assert np.array_equal(data.shape, (size[0], size[1], 3))

    return data


def image_to_renderable_metadata(pil_img, extra_metadata={}):

    data = np.array(pil_img).tolist()
    h = len(data)
    w = len(data[0])

    metadata = {
        "display_renderer": "ImageRenderer",
        "display_data": data,
        "h": h,
        "w": w
    }
    metadata.update(extra_metadata)

    return json.dumps(metadata)


def image_file_to_label(filename):
    return os.path.split(os.path.dirname(filename))[1]


def image_file_to_id(filename):
        # Drop the directory name
    filename = os.path.basename(filename)
    # Drop the .ico / .png suffix
    filename = os.path.splitext(filename)[0]
    return filename


def build_dataset(f, images, labels, metadata):

    f.create_dataset("x", data=np.asarray(images))
    f.create_dataset("metadata", (len(metadata),), dtype=VARLEN_STR)
    f.create_dataset("y", (len(metadata),), dtype=VARLEN_STR)

    f['metadata'][:] = metadata
    f['y'][:] = labels


class KerasBatchPreprocessorWrapper(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, x):
        o = self.function(np.array([x], dtype=np.float32))[0]
        return o

    def get_config(self):
        return {'class_name': self.__class__.__name__, 'config': {}}


class MobileNetWrapper(KerasBatchPreprocessorWrapper):
    def __init__(self):
        super(MobileNetWrapper,
              self).__init__(function=mobilenet_batch_preprocessor)


register_custom_object("MobileNetPreprocessor", MobileNetWrapper)


class MobileNetV2Wrapper(KerasBatchPreprocessorWrapper):
    def __init__(self):
        super(MobileNetV2Wrapper,
              self).__init__(function=mobilenet_v2_batch_preprocessor)


register_custom_object("MobileNetV2Preprocessor", MobileNetV2Wrapper)


class ResizeAndNormalize(MoiraiPreprocessor):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img):
        if img is None:
            return None

        shape = img.shape
        if shape[0] != self.h or shape[1] != self.w:
            img = Image.fromarray(img)
            img = img.resize((self.h, self.w))
            img = np.array(img)

        if shape[-1] == 4:
            img = img[..., :3]
        elif shape[-1] == 1:
            img = np.concatenate([img[0], img[0], img[0]])

        return normalize_image(img)


class Resize32x32(ResizeAndNormalize):
    def __init__(self):
        super(Resize32x32, self).__init__(32, 32)


class Resize48x48(ResizeAndNormalize):
    def __init__(self):
        super(Resize48x48, self).__init__(48, 48)


class Resize16x16(ResizeAndNormalize):
    def __init__(self):
        super(Resize16x16, self).__init__(16, 16)


class Resize192x192(ResizeAndNormalize):
    def __init__(self):
        super(Resize192x192, self).__init__(192, 192)


register_custom_object("ResizeAndNormalize", ResizeAndNormalize)
register_custom_object("Resize192x192", Resize192x192)
register_custom_object("Resize32x32", Resize32x32)
register_custom_object("Resize16x16", Resize16x16)
register_custom_object("Resize48x48", Resize48x48)
