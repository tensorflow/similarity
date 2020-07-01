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

import base64
import os
from io import BytesIO

import imageio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def base64_to_numpy(data):
    """Convert data from base64 encoding to numpy

    Args:
        data (String): The base64 encoded data string of image

    Returns:
        np.array: either [H x W x 3] or [H x W] depending on the input data
    """
    # the data is given with "header,content" format, we want the content
    encoded_image = data.split(",")[1]
    encoded_bytes = base64.b64decode(encoded_image)

    with open("temp.png", "wb") as f:
        f.write(encoded_bytes)

    # read the data from temp image file
    result = imageio.imread('temp.png')
    result = np.array(result)
    os.remove('temp.png')
    return result


def is_rgb(images):
    return len(images.shape) == 4


def beautify_grayscale(images):
    max_value = np.max(images)
    return max_value - images


def figure_to_src(figure):
    io_bytes = BytesIO()
    figure.savefig(io_bytes, format='png', bbox_inches='tight')
    io_bytes.seek(0)
    pic_hash = base64.b64encode(io_bytes.read())
    pic_hash_str = pic_hash.decode("utf-8")
    src = "data:image/png;base64," + pic_hash_str
    plt.close(figure)
    return src


def read_image_dataset_targets(directory, is_rgb, size, dict_key):
    """ Reads the target images from specified directory.
    """

    targets_directory = directory
    image_files = os.listdir(targets_directory)
    x_targets = [None] * len(image_files)
    y_targets = [None] * len(image_files)
    for i, image_file in enumerate(image_files):
        image_path = targets_directory + image_file
        if is_rgb:
            image_data = imageio.imread(image_path)[:, :, :3]
        else:
            image_data = imageio.imread(image_path)

        x_targets[i] = image_data
        # the label for each target is the name of the file without the
        # extension
        label = image_file.split('.')[0]
        y_targets[i] = label

    if is_rgb:
        # turn the target images into grayscale
        x_targets = np.asarray(x_targets)
        x_targets = tf.image.rgb_to_grayscale(x_targets).numpy()
        # normalize and map white (empty) pixels to 0 instead of max
        x_targets = (np.max(x_targets) - x_targets) / np.max(x_targets)
        
    else:
        x_targets = np.array(x_targets)
        x_targets = np.max(x_targets) - x_targets
        x_targets = x_targets / np.max(x_targets)
        x_targets = x_targets[..., np.newaxis]
        y_targets = np.array(y_targets)

    x_targets = tf.image.resize(x_targets, (size, size))
    packaged_x_targets = {dict_key: x_targets}

    return packaged_x_targets, y_targets


    
