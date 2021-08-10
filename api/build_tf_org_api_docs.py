# Lint as: python3
# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import tensorflow_privacy
"""Script to generate api_docs for TensorFlow Similarity.

$ pip install git+https://github.com/tensorflow/docs
$ python build_tf_org_api_docs.py --output_dir=/tmp/tfsim
"""

import os

from absl import app
from absl import flags

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api
from tensorflow_docs.api_generator import utils

import tensorflow_similarity as tfsim

import tensorflow_similarity.callbacks
import tensorflow_similarity.distance_metrics
import tensorflow_similarity.distances
import tensorflow_similarity.evaluators
import tensorflow_similarity.indexer
import tensorflow_similarity.losses
import tensorflow_similarity.matchers
import tensorflow_similarity.metrics
import tensorflow_similarity.models
import tensorflow_similarity.samplers
import tensorflow_similarity.tables
import tensorflow_similarity.visualization


flags.DEFINE_string('output_dir', '/tmp/tfsim',
                    'Where to output the docs.')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/similarity/blob/main/tensorflow_similarity',
    'The url prefix for links to code.')
flags.DEFINE_string('site_path', 'similarity/api_docs/python/',
                    'The location of the doc setin the site.')
flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files.')
flags.DEFINE_bool('gen_report', False,
                  ('Generate an API report containing the health of the'
                   'docstrings of the public API.'))

FLAGS = flags.FLAGS

PROJECT_SHORT_NAME = 'tfsim'
PROJECT_FULL_NAME = 'TensorFlow Similarity'


def _hide_layer_and_module_methods():
  """Hide methods and properties defined in the base classes of keras layers."""
  # __dict__ only sees attributes defined in *this* class, not on parent classes
  # Needed to ignore redudant subclass documentation
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())
  model_contents = list(tf.keras.Model.__dict__.items())
  module_contents = list(tf.Module.__dict__.items())
  optimizer_contents = list(tf.compat.v1.train.Optimizer.__dict__.items())

  for name, obj in model_contents + layer_contents + module_contents + optimizer_contents:

    if name == '__init__':
      continue

    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass


def gen_api_docs():
  """Generates api docs for the tensorflow docs package."""
  output_dir = FLAGS.output_dir

  _hide_layer_and_module_methods()
  doc_generator = generate_lib.DocGenerator(
      root_title=PROJECT_FULL_NAME,
      py_modules=[(PROJECT_SHORT_NAME, tfsim)],
      base_dir=os.path.dirname(tfsim.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      site_path=FLAGS.site_path,
      search_hints=FLAGS.search_hints,
      # This callback cleans up a lot of aliases caused by internal imports.
      callbacks=[public_api.local_definitions_filter])

  doc_generator.build(output_dir)
  print('Output docs to: ', output_dir)


def main(_):
  gen_api_docs()


if __name__ == '__main__':
  app.run(main)
