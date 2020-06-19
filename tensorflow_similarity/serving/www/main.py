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

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from matplotlib import pyplot as plt
from tensorflow_similarity.api.engine.simhash import SimHash

from tensorflow_similarity.serving.www.explain import Explainer
from tensorflow_similarity.serving.www.utils import (base64_to_numpy, beautify_grayscale, figure_to_src, is_rgb,
                   read_emoji_targets, read_mnist_targets)


class VueFlask(Flask):
    # Custom Flask because the default does not work with Vue.js
    jinja_options = Flask.jinja_options.copy()
    jinja_options.update(dict(
        variable_start_string='%%',
        # Default is '{{', I'm changing this because Vue.js uses '{{' / '}}'
        variable_end_string='%%',
    ))


app = VueFlask(__name__)

# Global storage to store objects (such as model) that should be computed
# only once every session
STORAGE = defaultdict(dict)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# width of the drawing board
WIDTH = 240
NUM_CHANNELS = 4
AREA = WIDTH ** 2


@app.route('/')
def index():
    # paths to models
    mnist_model_path = "saved_models/mnist_model.h5"
    emoji_model_path = "saved_models/emoji_model.h5"

    # load models
    mnist_model = tf.keras.models.load_model(mnist_model_path, custom_objects={'tf': tf})
    emoji_model = tf.keras.models.load_model(emoji_model_path, custom_objects={'tf': tf})

    # initialize Explainers
    mnist_explainer = Explainer(mnist_model.tower_model)
    emoji_explainer = Explainer(emoji_model.tower_model)

    # read in target data for mnist and emoji datasets
    mnist_x_targets, mnist_y_targets = read_mnist_targets()
    emoji_x_targets, emoji_y_targets = read_emoji_targets()

    # compute database with targets
    mnist_model.build_database(
        mnist_x_targets, mnist_y_targets)
    mnist_x_targets = mnist_x_targets["example"]

    emoji_model.build_database(emoji_x_targets, emoji_y_targets)
    emoji_x_targets = emoji_x_targets["image"]

    # store mnist model and data
    mnist_storage = STORAGE["mnist"]
    mnist_storage["model"] = mnist_model
    mnist_storage["explainer"] = mnist_explainer
    mnist_storage["x_targets"] = mnist_x_targets
    mnist_storage["y_targets"] = mnist_y_targets
    mnist_storage["size"] = 28

    # store emoji model and data

    emoji_storage = STORAGE["emoji"]
    emoji_storage["model"] = emoji_model
    emoji_storage["explainer"] = emoji_explainer
    emoji_storage["x_targets"] = emoji_x_targets
    emoji_storage["y_targets"] = emoji_y_targets
    emoji_storage["size"] = 32

    return render_template("index.html")


@app.route('/distances', methods=['POST'])
def get_distances():
    response_object = {'status': 'success'}

    dataset = request.get_json().get('dataset')
    storage = STORAGE[dataset]
    model = storage["model"]
    explainer = storage["explainer"]
    x_targets = storage["x_targets"]
    y_targets = storage["y_targets"]
    num_neighbors = len(y_targets)

    dictionary_key = "example"
    if dataset == "emoji":
        dictionary_key = "image"

    images_data = processing_request(request)

    x_test = {dictionary_key: images_data}

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

        # explainability
        original_images = np.array([x_target])
        heat_maps = explainer.explain(original_images)
        figure = explainer.render(original_images, heat_maps)
        explain_src = figure_to_src(figure)
        neighbor_explain_srcs.append(explain_src)

        distance = round(neighbor.distance, 3)
        # put it in an dictionary
        neighbor_obj = {"label": label, "distance": distance}
        result_data.append(neighbor_obj)
        if distance < smallest_distance:
            predicted_label = label
            smallest_distance = distance

    # explainability
    original_images = x_test[dictionary_key]
    heat_maps = explainer.explain(original_images)
    figure = explainer.render(original_images, heat_maps)

    explain_src = figure_to_src(figure)

    response_object["neighbors"] = result_data
    response_object["predicted_label"] = predicted_label
    response_object["explain_src"] = explain_src
    response_object["neighbor_explain_srcs"] = neighbor_explain_srcs

    original_image = beautify_grayscale(original_images[0])
    original_image = np.squeeze(original_image)
    figure, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(original_image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    original_img_src = figure_to_src(figure)
    response_object["original_img_src"] = original_img_src
    return jsonify(response_object)


def processing_request(request):
    """Process the request to compute distances from the frontend
       and return the numpy array of input images. Currently only supports
       uploading one image."""

    request_json = request.get_json()

    image_data = request_json.get('data')
    dataset = request_json.get('dataset')

    # receieve an uploaded image
    if isinstance(image_data, str):
        uploaded_image_arr = base64_to_numpy(image_data)

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
        size = STORAGE[dataset]["size"]

        # flatten the dictionary into an array
        # the dictionary given from frontend if formatted
        # {"203": 20, "30", 3, ...} thus we want to turn the key into int
        # and insert the value into the array
        for key in image_data:
            int_key = int(key)
            value = image_data[key]
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
