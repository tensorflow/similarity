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

from absl import app, flags
from flask import Flask, request
import json
from tensorflow_similarity import common_flags
from tensorflow_similarity.utils.config_utils import load_custom_configuration, json_dict_to_moirai_obj, load_json_configuration_file
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow_similarity.ui2 import server_lib
from tensorflow_similarity.ui2.flask_routes import configure_flask
from tensorflow_similarity.ui2 import input_handler
import tensorflow as tf
from tensorflow.python import ConfigProto, Session

flags.DEFINE_string("host", "0.0.0.0", "Address to listen upon.")
flags.DEFINE_integer("port", 5010, "Port to run the server on.")
flags.DEFINE_string("config", None, "JSON Config used to serve.")

FLAGS = flags.FLAGS


def configure_tensorflow():
    config = ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    tf.keras.backend.set_session(
        sess)  # set this TensorFlow session as the default session for Keras
    tf.keras.backend.set_learning_phase(0)


def ui_main(args):
    config = load_json_configuration_file(FLAGS.config)

    if 'custom_configuration_module' in config:
        load_custom_configuration(config['custom_configuration_module'])

    server_lib.configure_server(config)

    flask_app = configure_flask()
    flask_app.run(host=FLAGS.host, port=FLAGS.port, debug=True)


if __name__ == '__main__':
    app.run(ui_main)
