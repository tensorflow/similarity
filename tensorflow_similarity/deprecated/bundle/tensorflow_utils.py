#!/usr/bin/python

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

"""Utility which builds a Moirai bundle, consisting of a model for generating
embeddings, and a set of precomputed embeddings for known points."""

import hashlib
import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.python import GraphDef


def get_input_ops(keras_model):
    return [node.op.name for node in keras_model.inputs]


def get_input_nodes(keras_model):
    return [node.op for node in keras_model.inputs]


def get_output_op(keras_model):
    return keras_model.output.op.name


def load_graph_from_buffer(graph, contents, input_nodes, output_node):

    graph_def = GraphDef()
    graph_def.ParseFromString(contents)
    import_prefix = hashlib.md5(contents).hexdigest()

    with graph.as_default():
        tf.import_graph_def(
            graph_def,
            return_elements=None,
            name=import_prefix,
        )

        input_dict = {}

        for node in input_nodes:
            input_dict[node] = "%s/%s:0" % (import_prefix, node)

        return (input_dict, "%s/%s:0" % (import_prefix, output_node))


def load_graph(model_file, input_node, output_node):
    with tf.io.gfile.GFile(model_file, "rb") as f:
        graph_def = GraphDef()
        return load_graph_from_buffer(f.read(), input_node, output_node)
