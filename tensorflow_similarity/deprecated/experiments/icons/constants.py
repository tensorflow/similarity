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

from icongenerator.data_utils import get_file
import uuid
import numpy as np


# Adjustable knobs.
NUM_GPUS = 1
ICON_SIZE = 48
AUTOENCODER_ICON_SIZE = 48
BATCH_SIZE = 64
NUM_GEN_ICONS_PER_EPOCH = 100000
MODEL_ID = str(uuid.uuid4())

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

AUGMENTATION_PARAMS_FULL = {
    "background_frequency": 0.5,
    "shape_frequency": 0.5,
    "min_logo_size": 0.50,
    "max_logo_size": 0.95,
    "use_flips": True,
    "occlusion": "light",  # None | "light" | "heavy"'
    "use_logo_color_shift": True,
    "blur": 1.0,
    "max_rotation": 45,
    "max_affine_transformations": 2,
    "use_background_color_shift": True,
    "use_shape_color_shift": True,
    "min_logo_alpha": 230
}
AUGMENTATION_PARAMS = {
    "background_frequency": 0.1,
    "shape_frequency": 0.1,
    "min_logo_size": 0.50,
    "max_logo_size": 0.95,
    "use_flips": True,
    "occlusion": "light",  # None | "light" | "heavy"'
    "use_logo_color_shift": True,
    "blur": 1.0,
    "max_rotation": 45,
    "max_affine_transformations": 1,
    "use_background_color_shift": True,
    "use_shape_color_shift": True,
    "min_logo_alpha": 230
}
SIMILARITY_EMBEDDING_SIZE = 128
VIZ_ICONS = [b'google_photos', b'tensorflow',
             b'google_maps', b'google_ai', b'google']

DATASET_FILENAME = get_file(
    "icons.h5",
    "https://firebasestorage.googleapis.com/v0/b/kerastuner-prod.appspot.com/o/icons.h5?alt=media&token=2051dcaa-8152-476b-977b-f06aa224b709")
