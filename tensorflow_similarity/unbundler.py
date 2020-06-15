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
# Utility which unbundles a Moirai bundle into separate files containing
# the tensorflow model, the target points, and the calibrated thresholds.

from tqdm import tqdm
import os
import tensorflow as tf
import tensorflow_similarity
from tensorflow_similarity.utils.config_utils import load_custom_configuration
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

flags.DEFINE_string('input_bundle', None, '')
flags.DEFINE_string('output_directory', None, '')

FLAGS = flags.FLAGS


def main(args):

    load_custom_configuration()

    bundle = BundleWrapper.load(FLAGS.input_bundle)

    graphdef_file = os.path.join(FLAGS.output_directory, "graphdef.pb")
    gfv_file = os.path.join(FLAGS.output_directory, "points.csv")
    thresholds_file = os.path.join(FLAGS.output_directory, "thresholds.csv")

    raw_graph_def = bundle.get_serialized_model()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(raw_graph_def)

    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()

        inputs = {}
        outputs = {}
        for node in bundle.get_input_nodes():
            inputs[node] = g.get_tensor_by_name("%s:0" % node)
        node = bundle.get_output_node()
        outputs[node] = g.get_tensor_by_name("%s:0" % node)

        signatures = {}
        signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
            inputs, outputs)
        builder = tf.saved_model.builder.SavedModelBuilder(
            FLAGS.output_directory)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING], signature_def_map=signatures)

        builder.save()

    with tf.io.gfile.GFile(graphdef_file, "w") as f:
        f.write(raw_graph_def)

    points = bundle.get_points()
    labels = bundle.get_labels()
    with tf.io.gfile.GFile(gfv_file, "w") as f:
        for p, l in tqdm(
                zip(points, labels), total=len(points), desc=gfv_file):
            f.write("%s,%s\n" % (l, ','.join([str(x) for x in p])))

    print("Writing thresholds to: ", thresholds_file)
    thresholds = bundle.get_global_thresholds()
    with tf.io.gfile.GFile(thresholds_file, "w") as f:
        for t, p in bundle.get_global_thresholds():
            f.write("%s,%s\n" % (t, p))


if __name__ == '__main__':
    app.run(main)
