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
import string

import imageio
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras

from tensorflow_similarity.serving.www.explain import Explainer
from tensorflow_similarity.serving.www.constants import IMDB_REVIEW_LENGTH

from matplotlib import pyplot as plt

import json
from tabulate import tabulate

def load_model(model_path):
    """ Load a tf similarity model

    Args:
        model_path (String): the path to the stored model

    Returns:
        model: the loaded tf model
        dictionary_key: the dictionary key of the dict that must be passed to the model
        explainer: Explainer instance which implements GradCam for explainability
    """
    # load model and initialize explainer
    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
    dictionary_key = model.layers[0].name
    if "IMDB" in model_path:
        explainer = None
    else:
        explainer = Explainer(model.tower_model)

    return model, dictionary_key, explainer


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

def get_imdb_dict():
    imdb = tf.keras.datasets.imdb
    word_index = imdb.get_word_index()
    word_index = {k: (v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    return word_index

def encode_review(text):
    """Convert a string to a numpy array for the pretrained IMDB model

    Args:
        text (String): The text to be converted to a numpy array

    Returns:
        (np.array): the numpy array representation of the string compatible with the model
    """
    word_index = get_imdb_dict()
    text = text.translate(string.punctuation)
    text = text.lower().split(" ")
    if len(text) > IMDB_REVIEW_LENGTH - 2:
        text = text[IMDB_REVIEW_LENGTH - 2:]
    for i in range (0, len(text)):
        text[i] = word_index.get(text[i], 2)
    text.insert(0, 1)
    
    while len(text) < IMDB_REVIEW_LENGTH:
        text.append(0)
    return np.asarray(text)


def decode_review(text):
    """Convert a string encoded as a numpy array back to its string representation

    Args:
        text (np.array): The text to be converted to a numpy array

    Returns:
        (String): the String representation of the np array compatible with the model
    """
    word_index = get_imdb_dict()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def read_text_dataset_targets(targets_directory, dict_key):
    """ Reads the target text files from specified directory.
    """
    text_files = os.listdir(targets_directory)
    x_targets = [None] * len(text_files)
    y_targets = [None] * len(text_files)
    for i, text_file in enumerate(text_files):
        text_path = targets_directory + text_file
        f = open(text_path, "r")
        text_data = f.read()
        f.close()
        text_data = encode_review(text_data)

        x_targets[i] = text_data
        # the label for each target is the name of the file without the
        # extension
        label = '1' if text_file.split('.')[0].split('-')[0] == "positive" else '0'
        y_targets[i] = label
    packaged_x_targets = {dict_key: np.asarray(x_targets)}

    return packaged_x_targets, y_targets



def read_image_dataset_targets(targets_directory, is_rgb, size, dict_key):
    """ Reads the target images from specified directory.
    """
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


    
