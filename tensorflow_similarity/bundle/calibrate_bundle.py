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
from absl import app, flags
import json
import tensorflow_similarity
from tensorflow_similarity.utils.config_utils import load_custom_configuration, load_json_configuration_file
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow_similarity.bundle.tensorflow_utils import *

import tensorflow as tf
from tensorflow_similarity.bundle.calibration_v2 import calibrate

flags.DEFINE_string('input_config', None, '')
flags.DEFINE_string('input_bundle', None, '')
flags.DEFINE_string('output_bundle', 'calibrated_bundle', '')
flags.DEFINE_boolean(
    'regenerate',
    False,
    "Regenerate the intermediate data, even if it already exists.")

FLAGS = flags.FLAGS


def main(args):
    load_custom_configuration()

    print("Loading bundle...")
    bundle = BundleWrapper.load(FLAGS.input_bundle)
    print("Loading config...")
    input_config = load_json_configuration_file(FLAGS.input_config)
    print("Calibrating...")
    calibrate(input_config, bundle)
    print("Writing new bundle...")
    bundle.write(FLAGS.output_bundle)


if __name__ == '__main__':
    app.run(main)
