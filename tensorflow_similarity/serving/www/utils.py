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

from tensorflow_similarity.serving.www.explain import Explainer

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

def read_config(config_file):
    with open(os.path.dirname(__file__) + "/" + config_file, 'r') as f:
        config = json.load(f)
    return config

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
    """Create a word to index mapping to convert plain text to a format that
       can be handled by a model
    
    Returns:
        word_index (dict): a dictionary of words mapping to their respective integer representation
    """
    imdb = tf.keras.datasets.imdb
    word_index = imdb.get_word_index()
    # shift all indeces up by 3 to add PAD START UNK and UNUSED tokens to dict
    word_index = {k: (v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    return word_index

def encode_review(text):
    """Convert a string to a numpy array compatible with the pretrained IMDB model

    Args:
        text (String): The text to be converted to a numpy array

    Returns:
        (np.array): the numpy array representation of the string compatible with the model
    """
    word_index = get_imdb_dict()
    # preprocess string by removing punctuation, converting to lower case and splitting into words 
    text_arr = text.translate(string.punctuation).lower().split(" ")
    IMDB_REVIEW_LENGTH = read_config("serving_config.json")["imdb"]["review_length"]
    if len(text_arr) > IMDB_REVIEW_LENGTH - 2:
        # if the review is longer than IMDB_REVIEW_LENGTH - 2 words, truncate to IMDB_REVIEW_LENGTH - 2 words 
        # to account for START and STOP tokens
        text_arr = text_arr[IMDB_REVIEW_LENGTH - 2:]
    for i, word in enumerate(text_arr):
        # replace every word in the array with its integer representation in the word_index or UNK 
        # if it isn't in the word_index
        text_arr[i] = word_index.get(word, 2)
    # insert the START token at the beginning of the array
    text_arr.insert(0, 1)
    # pad the array to the correct length with PAD tokens
    num_zero_filling = IMDB_REVIEW_LENGTH - len(text_arr)
    text_arr.extend([0] * num_zero_filling)
    return np.asarray(text_arr)


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

    Args: 
        targets_directory (String): the directory from which targets should be read
        dict_key (String): the key that is used by a model to access targets in a dict

    Returns:
        packaged_x_targets (dict): a dict containting the target text files encoded as np arrays
        y_targets (list): the corresponsing labels for the x_targets
    """
    text_files = os.listdir(targets_directory)
    x_targets = [None] * len(text_files)
    y_targets = [None] * len(text_files)
    for i, text_file in enumerate(text_files):
        # open each text file, read the data and encode it as a np array
        text_path = targets_directory + text_file
        f = open(text_path, "r")
        text_data = f.read()
        f.close()
        text_data = encode_review(text_data)

        x_targets[i] = text_data
        # the label for each target is 1 if the file name starts with positive
        # and 0 otherwise
        if text_file.split('.')[0].split('-')[0] == "positive":
            label = '1'
        else:
            label = '0'
        y_targets[i] = label
    packaged_x_targets = {dict_key: np.asarray(x_targets)}

    return packaged_x_targets, y_targets



def read_image_dataset_targets(targets_directory, is_rgb, size, dict_key):
    """ Reads the target images from specified directory.
        
    Args: 
        targets_directory (String): the directory from which targets should be read
        is_rgb (bool): are images in rgb or grayscale
        size (integer): the dimension of the image (size * size)
        dict_key (String): the key that is used by a model to access targets in a dict

    Returns:
        packaged_x_targets (dict): a dict containting the target image files encoded as np arrays
        y_targets (list): the corresponsing labels for the x_targets
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


    
