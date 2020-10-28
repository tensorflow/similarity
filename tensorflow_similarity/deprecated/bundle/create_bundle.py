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

from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
import tempfile
import os
from absl import app, flags
import json
import tensorflow_similarity
from tensorflow_similarity import common_flags
from tensorflow_similarity.model import Moirai
from tensorflow_similarity.utils.config_utils import load_custom_configuration, deserialize_moirai_object, serialize_moirai_object, load_json_configuration_file
from tensorflow_similarity.utils.config_utils import json_dict_to_moirai_obj, load_json_configuration_file
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow_similarity.dataset import Dataset, DatasetConfig
from tensorflow_similarity.preprocessing import images
import numpy as np
from tqdm import tqdm
from six.moves import range
from tensorflow.python import ConfigProto, Session
from tensorflow.keras.backend import set_session, clear_session, set_learning_phase
from tensorflow_similarity.bundle.tensorflow_utils import get_input_ops, get_output_op
from tensorflow_similarity.bundle.calibration_v2 import calibrate
import tensorflow as tf
import tensorflow.keras.utils as tf_utils

flags.DEFINE_string('input_config', None, '')
flags.DEFINE_string('model_file', None, '')
flags.DEFINE_string('output_file', 'moirai_bundle', '')

flags.DEFINE_integer('workers', 60, '')
flags.DEFINE_integer('K', 10, '')

flags.DEFINE_integer(
    'target_augments', 100,
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


def _get_output_tensor_names_from_savedmodel(model, saved_model_path):
    """ Looks in the default_serving signature def of the saved model to
    determine the output tensor names for the given model.
    """
    saved_model_pb_file = os.path.join(saved_model_path, "saved_model.pb")

    with tf.io.gfile.GFile(saved_model_pb_file, "rb") as f:
        graph_bytes = f.read()

    sm = SavedModel()
    sm.ParseFromString(graph_bytes)

    name_map = {}

    for meta_graph in sm.meta_graphs:
        sig_def = meta_graph.signature_def["serving_default"]
        for name, tensor in sig_def.outputs.items():
            tensor_name = tensor.name
            # Drop the :0 suffix.
            tensor_name = tensor_name.split(":")[0]
            name_map[name] = tensor_name

    outputs = []
    for output_name in model.output_names:
        outputs.append(name_map[output_name])
    return outputs


def create_bundle(input_config, model_file, output_file):
    config = load_json_configuration_file(input_config)

    print("Load")
    model = tf.keras.models.load_model(model_file, compile=False)
    model.compile(loss="mse", optimizer="adam")

    input_nodes = get_input_ops(model)
    print("Nodes", input_nodes)

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

    if tf.io.gfile.exists(tmp_dir):
        tf.io.gfile.rmtree(tmp_dir)
    tf.io.gfile.makedirs(tmp_dir)

    tmp_dir = tempfile.mkdtemp()
    saved_model_path = "%s/tf.savedmodel.pb" % tmp_dir
    tmp_dir2 = tempfile.mkdtemp()
    frozen_model_path = "%s/tf.frozenmodel.pb" % tmp_dir2

    tf_utils.save_model(model, saved_model_path,
                        export_type="tf", tmp_path=tmp_dir)
    tf_utils.save_model(model, frozen_model_path,
                        export_type="tf_frozen", tmp_path=tmp_dir2)

    examples = []
    labels = []
    metadata = []

    for i in range(len(dataset)):
        item = dataset.get_item(i, augment=False, preprocess=True)
        examples.append(item.feature_dictionary)
        label = de_numpy(item.label)
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        labels.append(label)
        metadatum = de_numpy(item.metadata)
        metadata.append(metadatum)

    input_nodes = get_input_ops(model)

    output_nodes = _get_output_tensor_names_from_savedmodel(
        model, saved_model_path)

    model = None

    with tf.io.gfile.GFile(frozen_model_path, "rb") as f:
        serialized_tf_model = f.read()

    bundle = BundleWrapper(
        serialized_tf_model, [[x] for x in range(len(labels))],
        labels,
        metadata,
        input_nodes=input_nodes,
        output_node=output_nodes[0])
    inference = bundle.get_inference()
    results = [x['embedding'] for x in inference.embed(examples)]

    bundle = BundleWrapper(
        serialized_tf_model,
        results,
        labels,
        metadata,
        input_nodes=input_nodes,
        output_node=output_nodes[0])

    moirai = None
    print("Writing model bundle to %s" % output_file)
    bundle.write(output_file)
    print("Done!")
    return bundle


def main(args):
    config = ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)
    set_learning_phase(0)

    load_custom_configuration()

    model_file = FLAGS.model_file
    if FLAGS.using_remote_filesystem:
        local_model_file = os.path.join(FLAGS.local_tmp_dir, "tmp_model")
        tf.io.gfile.copy(model_file, local_model_file)
        model_file = local_model_file

    create_bundle(FLAGS.input_config, model_file, FLAGS.output_file)


if __name__ == '__main__':
    app.run(main)
