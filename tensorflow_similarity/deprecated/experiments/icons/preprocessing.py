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
import copy
from tensorflow_similarity.experiments.icons.constants import *


def mobilenet_preprocessor(x):
    out = copy.copy(x)

    img = x["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.resize((224, 224))
    img = np.array(img)
    img = img[..., 0:3]
    out["image"] = img / 255.0

    return out


def image_preprocessor(x):
    out = copy.copy(x)

    img = x["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.resize((ICON_SIZE, ICON_SIZE))
    img = np.array(img)
    img = img[..., 0:3]
    out["image"] = img / 255.0

    return out


def wdsr_image_preprocessor(x):
    out = {}
    img = x["image"]
    if not isinstance(img, Image.Image):
        try:
            img = Image.fromarray(img)
        except BaseException:
            raise ValueError(
                "Could not convert image from item with shape %s" % str(
                    np.shape(img)))
        img = img.convert("RGB")

    img = img.resize((ICON_SIZE, ICON_SIZE))
    img = np.array(img)
    img = img[..., 0:3]
    out["image"] = img
    return out


def batch_image_preprocessor(x):
    out = []
    for img in x:
        img = Image.fromarray(img).resize((ICON_SIZE, ICON_SIZE))
        img = np.array(img)
        out.append(np.array(img)[:, :, 0:3] / 255.0)

    return np.array(out)
