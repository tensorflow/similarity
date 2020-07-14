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

from collections import defaultdict

import os
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from tensorflow_similarity.api.engine.simhash import SimHash

from tensorflow_similarity.serving.www.explain import Explainer
from tensorflow_similarity.serving.www.utils import (load_model, base64_to_numpy, beautify_grayscale, figure_to_src, is_rgb,
                 read_image_dataset_targets, read_text_dataset_targets, encode_review, decode_review)


class VueFlask(Flask):
    # Custom Flask because the default does not work with Vue.js
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        variable_start_string='%%',
        # Default is '{{', I'm changing this because Vue.js uses '{{' / '}}'
        variable_end_string='%%',
    ))


app = VueFlask(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# width of the drawing board
WIDTH = 240
NUM_CHANNELS = 4
AREA = WIDTH ** 2


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/distances', methods=['POST'])
def get_distances():
    dataset = request.get_json().get('dataset')
    response_object = {'status': 'success'}
    base_path = os.path.dirname(__file__)

    if dataset == "mnist":
        # pretrained mnist model
        model_path = base_path + "/saved_models/mnist_model.h5"
        targets_directory = base_path + "/static/images/mnist_targets/"
        is_rgb = False
        size = 28
    elif dataset == "emoji":
        # pretrained emoji model
        model_path = base_path + "/saved_models/emoji_model.h5"
        targets_directory = base_path + "/static/images/emoji_targets/"
        is_rgb = True
        size = 32
    elif dataset == "imdb":
        # pretrained imdb model
        targets_directory = base_path + "/static/text/imdb_targets/"
        model_path = base_path + "/saved_models/IMDB_model.h5"
        size = -1

    # load model
    model, dictionary_key, explainer = load_model(model_path)

    # read in target data
    if dataset == "imdb":
        x_targets, y_targets = read_text_dataset_targets(targets_directory, dictionary_key)
    else:
        x_targets, y_targets = read_image_dataset_targets(targets_directory, is_rgb, size, dictionary_key)
    
    num_neighbors = len(y_targets)

    # compute target database
    model.build_database(x_targets, y_targets)
    x_targets = x_targets[dictionary_key]

    request_data = processing_request(request, size)
    x_test = {dictionary_key: request_data}

    # get neighbors
    neighbors = model.neighbors(x_test, num_neighbors)[0]
    # the result will be a dictionary where the key is the label and
    # the value is the distance
    result_data = list()
    smallest_distance = 10
    predicted_label = None
    neighbor_explain_srcs = []
    for neighbor in neighbors:
        label = str(neighbor.label)
        index = neighbor.index
        x_target = x_targets[index]

        # put it in an dictionary
        neighbor_obj = {"label": label}

        # explainability
        if explainer is not None:
            original_images = np.array([x_target])
            heat_maps = explainer.explain(original_images)
            figure = explainer.render(original_images, heat_maps)
            explain_src = figure_to_src(figure)
            neighbor_explain_srcs.append(explain_src)
        else:
            original_images = request_data
            explain_src = []
            neighbor_obj["text"] = decode_review(x_target)

        distance = round(neighbor.distance, 3)
        
        result_data.append(neighbor_obj)
        if distance < smallest_distance:
            predicted_label = label
            smallest_distance = distance

        neighbor_obj["distance"] = distance
    # explainability
    if explainer is not None:
        original_images = x_test[dictionary_key]
        heat_maps = explainer.explain(original_images)
        figure = explainer.render(original_images, heat_maps)

        explain_src = figure_to_src(figure)
    else:
        original_text = x_test[dictionary_key][0].tolist()
        explain_src = original_text

    response_object["neighbors"] = result_data
    response_object["predicted_label"] = predicted_label
    response_object["explain_src"] = explain_src
    response_object["neighbor_explain_srcs"] = neighbor_explain_srcs

    if dataset == "imdb":
        response_object["original_text"] = request.get_json().get('data')
    else:
        original_image = beautify_grayscale(original_images[0])
        original_image = np.squeeze(original_image)
        figure, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(original_image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        original_img_src = figure_to_src(figure)
        response_object["original_img_src"] = original_img_src
    return jsonify(response_object)


def processing_request(request, size):
    """Process the request to compute distances from the frontend
       and return the numpy array of input images. Currently only supports
       uploading one image."""

    request_json = request.get_json()

    request_data = request_json.get('data')
    dataset = request_json.get('dataset')

    if size == -1:
        result = np.asarray([encode_review(request_data)])
    else:
        # receieve an uploaded image
        if isinstance(request_data, str):
            uploaded_image_arr = base64_to_numpy(request_data)

            # the uploaded image is grayscale, need to extend a new dimension
            if len(uploaded_image_arr.shape) == 2:
                uploaded_image_arr = uploaded_image_arr[..., np.newaxis]
                uploaded_image_arr = beautify_grayscale(uploaded_image_arr)
                uploaded_image_arr = np.divide(uploaded_image_arr, 255.0)
            else:
                if uploaded_image_arr.shape[-1] == 4:
                    uploaded_image_arr = uploaded_image_arr[:, :, :3]
                    uploaded_image_arr = tf.image.rgb_to_grayscale(
                        uploaded_image_arr).numpy()

            # package the data into tf-similarity comptabile input
            result = np.asarray([uploaded_image_arr])
        else:
            # receive a drawing
            data_arr = np.zeros(AREA * NUM_CHANNELS)

            # flatten the dictionary into an array
            # the dictionary given from frontend if formatted
            # {"203": 20, "30", 3, ...} thus we want to turn the key into int
            # and insert the value into the array
            for key in request_data:
                int_key = int(key)
                value = request_data[key]
                data_arr[int_key] = value

            # every 4 elements in the data_arr specify the color of a pixel
            # (r,g,b,a), since we only allow the users to draw in one color, it
            # will be in grayscale and thus we can reconstruct the grayscale of each
            # pixel by reading the rgb for each pixel and convert them into
            # grayscale

            flatten_arr = np.zeros(AREA)
            for i in range(len(flatten_arr)):
                index = i * 4
                red = data_arr[index]
                green = data_arr[index + 1]
                blue = data_arr[index + 2]
                gray = 0.2125 * red + 0.7154 * green + 0.0721 * blue
                flatten_arr[i] = gray

            # reshape the drawing into 2D and resize to appropriate size to fit
            # into the model
            arr = np.reshape(flatten_arr, (WIDTH, WIDTH))
            # normalize
            arr = arr / np.amax(arr)
            # turn into tf tensor and resize
            arr_tensor = tf.constant(arr[..., np.newaxis])
            arr = tf.image.resize(arr_tensor, (size, size))

            # package the data into tf-similarity comptabile input
            result = np.asarray([arr])

    return result


if __name__ == '__main__':
    app.run(debug=True)
