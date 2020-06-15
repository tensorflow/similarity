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
from logging.config import dictConfig
import tensorflow as tf
from flask import Flask, request, redirect
from tensorflow_similarity.ui2 import server_lib
import os

DIR = os.path.dirname(__file__)
flags.DEFINE_string("template_directory", os.path.join(DIR, "templates"),
                    "Directory where HTML templates are stored.")

FLAGS = flags.FLAGS


def configure_flask():
    flask_app = Flask(__name__, template_folder=FLAGS.template_directory)

    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        }
    })

    app = Flask(__name__)

    flask_app.config.from_object(__name__)

    @flask_app.route('/ui/search', methods=["POST", "GET"])
    def search_ui():
        return server_lib.search_ui(request)

    return flask_app
