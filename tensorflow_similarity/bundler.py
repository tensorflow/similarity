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
import json
import tensorflow_similarity
from tensorflow_similarity import common_flags
from tensorflow_similarity.model import Moirai
from tensorflow_similarity.utils.config_utils import load_custom_configuration, deserialize_moirai_object, serialize_moirai_object, load_json_configuration_file
from tensorflow_similarity.utils.config_utils import json_dict_to_moirai_obj, load_json_configuration_file
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow_similarity.dataset import Dataset, DatasetConfig
import numpy as np
from tqdm import tqdm
from tensorflow_similarity.bundle.tensorflow_utils import *
from six.moves import range
import tensorflow as tf
from tensorflow.keras.backend import set_session, clear_session, set_learning_phase
from tensorflow_similarity.bundle.calibration_v2 import calibrate

flags.DEFINE_string('input_config', None, '')
flags.DEFINE_string('model_file', None, '')
flags.DEFINE_string('config_file', None, '')
flags.DEFINE_string('output_file', 'moirai_bundle', '')

flags.DEFINE_integer('workers', 60, '')
flags.DEFINE_integer('K', 10, '')

flags.DEFINE_boolean(
    'augmentation_based_calibration',
    False,
    'If true, use augmented versions of the target dataset and an inference dataset'
    ' to generate precision/threshold data, and use this to calibrate the model.')
flags.DEFINE_integer(
    'target_augments',
    100,
    'Number of augmentations per target to use for calibration.')

FLAGS = flags.FLAGS


def get_optional_config(config, name, raw=False):
    if name not in config:
        return None
    cfg = config[name]
    if raw:
        return cfg
    return json_dict_to_moirai_obj(cfg)


def de_numpy(x):
    if isinstance(x, np.generic):
        x = x.item()
    return x


def create_bundle(input_config,
                  config_file,
                  model_file,
                  output_file,
                  dep_module=None):
    config = load_json_configuration_file(input_config)

    model = tf.keras.models.load_model(model_file)

    reader = json_dict_to_moirai_obj(config['targets_reader'])
    common_dataset_config = get_optional_config(config, 'common_dataset', None)
    dataset_config = get_optional_config(config, 'targets_dataset', None)
    calibration_dataset_config = get_optional_config(
        config, 'calibration_dataset', None)
    inference_dataset_config = get_optional_config(config, 'inference_dataset',
                                                   None)

    if not dataset_config:
        dataset_config = common_dataset_config
    if not inference_dataset_config:
        inference_dataset_config = common_dataset_config
    if not calibration_dataset_config:
        calibration_dataset_config = common_dataset_config

    data = reader.read()
    dataset = Dataset(data, dataset_config)

    tmp_dir = "%s.dir" % output_file

    if tf.io.gfile.Exists(tmp_dir):
        tf.io.gfile.DeleteRecursively(tmp_dir)
    tf.io.gfile.MakeDirs(tmp_dir)

    saved_model_path = "%s/tf.savedmodel" % tmp_dir
    frozen_model_path = "%s/tf.savedmodel.pb" % tmp_dir
    transformed_model_path = "%s/tf.transformed.pb" % tmp_dir

    convert_to_tf_saved_model(model, saved_model_path)

    freeze(model, saved_model_path, frozen_model_path)
    transform_graph(model, frozen_model_path, transformed_model_path)
    final_path = transformed_model_path

    examples = []
    labels = []
    metadata = []

    for item in dataset:
        examples.append(item.feature_dictionary)
        labels.append(de_numpy(item.label))
        metadata.append(de_numpy(item.metadata))

    input_nodes = get_input_ops(model)
    output_node = get_output_op(model)

    model = None
    with tf.io.gfile.GFile(final_path, "rb") as f:
        serialized_tf_model = f.read()

    bundle = BundleWrapper(
        serialized_tf_model,
        [[x] for x in range(len(labels))],
        labels,
        metadata,
        input_nodes=input_nodes,
        output_node=output_node)
    inference = bundle.get_inference()
    results = [x['embedding'] for x in inference.embed(examples)]
    bundle = BundleWrapper(
        serialized_tf_model,
        results,
        labels,
        metadata,
        input_nodes=input_nodes,
        output_node=output_node)

    moirai = None
    print("Writing model bundle to %s" % output_file)
    bundle.write(output_file)
    return bundle


def main(args):
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(
        sess)  # set this TensorFlow session as the default session for Keras
    set_learning_phase(0)

    load_custom_configuration()

    model_file = FLAGS.model_file
    if FLAGS.using_remote_filesystem:
        local_model_file = os.path.join(FLAGS.local_tmp_dir, "tmp_model")
        tf.io.gfile.Copy(model_file, local_model_file)
        model_file = local_model_file

    uncal_output_file = FLAGS.output_file
    if FLAGS.augmentation_based_calibration:
        uncal_output_file = "%s.precalibration" % uncal_output_file

    create_bundle(
        FLAGS.input_config,
        FLAGS.config_file,
        model_file,
        uncal_output_file,
        dep_module=FLAGS.custom_dependency_module)

    if FLAGS.augmentation_based_calibration:
        bundle = BundleWrapper.load(uncal_output_file)
        input_config = load_json_configuration_file(FLAGS.input_config)
        calibrate(input_config, bundle)
        bundle.write(FLAGS.output_file)


if __name__ == '__main__':
    app.run(main)
