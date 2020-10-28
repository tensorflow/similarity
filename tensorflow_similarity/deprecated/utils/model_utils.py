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

import collections

import numpy as np
import six
import tensorflow as tf
from tensorflow.keras.layers import Input, InputLayer
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib


def get_input_layers(model):
    output = []
    for layer in model.layers:
        if isinstance(layer, InputLayer):
            output.append(layer)
    return output


def get_input_names(model):
    layers = get_input_layers(model)
    return [layer.name for layer in layers]


def index_layers(input_layers):
    output = {}
    for layer in input_layers:
        output[layer.name] = layer
    return output


def index_inputs(model):
    input_layers = get_input_layers(model)
    return index_layers(input_layers)


def clone_model_input(layer, prefix=""):
    name = "%s%s" % (prefix, layer.name)
    dtype = layer.dtype

    shape = layer.input_shape
    # TF 1.12, or 2.x
    if isinstance(shape, list):
        assert len(
            shape
        ) == 1, "Don't know how to handle multiple shapes for an input."
        shape = shape[0]

    shape_without_batch_size = shape[1:]
    new_input = Input(name=name, shape=shape_without_batch_size, dtype=dtype)
    return name, new_input


def clone_model_inputs(model, prefix=""):
    layers = get_input_layers(model)
    layer_inputs = []
    names = []
    for layer in layers:
        layer_input, name = clone_model_input(layer, prefix)
        layer_inputs.append(layer_input)
        names.append(name)
    return layer_inputs, names


def clone_task_inputs(task, prefix=""):
    layers = task.get_input_layers()
    names = task.get_input_names()

    new_layer_inputs = []
    for name, layer in zip(names, layers):
        name = "%s%s" % (prefix, name)
        names.append(name)
        dtype = layer.dtype

        # Drop the batch-size from the shape
        shape = layer.shape.as_list()[1:]
        new_input = Input(name=name, shape=shape, dtype=dtype)
        new_layer_inputs.append(new_input)
    return new_layer_inputs, names


def input_shape(layer):
    shape = layer.input_shape.as_list()
    shape = shape[1:]
    return shape


def compute_size(shape):
    print(shape)
    if isinstance(shape, tf.Tensor):
        shape = shape.shape.as_list()
    print(shape)
    if not isinstance(shape, list):
        shape = list(shape)
    print(shape)
    if shape[0] is None:
        shape = shape[1:]
    print(shape)

    output_size = 1
    for dim in shape:
        output_size *= dim
    return output_size


def layer_shape(layer):
    return layer.shape.as_list()[1:]


def get_devices():
    local_device_protos = device_lib.list_local_devices()
    output = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(output) == 0:
        output = ["/cpu:0"]
        return output

    return output


def training_data_to_example_list(dict_of_features):
    """Converts a dictionary of np.arrays to a list of dictionaries, where
    each item in the list is a dictionary of feature_name to the value for that
    example.

    Args:
        dict_of_features (dict{str, np.array}): Dictionary of feature name to
        np.array

    Returns:
        list[dict{str, np.array}]: List of feature examples, as dictionaries of
        features.
    """
    random_key = [k for k in dict_of_features.keys()][0]
    size = len(dict_of_features[random_key])

    examples = []
    for i in range(size):
        examples.append({})

    for k, v in six.iteritems(dict_of_features):
        for i in range(len(v)):
            examples[i][k] = v[i]

    return examples


def example_list_to_training_data(example_list):
    output = collections.defaultdict(list)
    for example in example_list:
        for k, v in six.iteritems(example):
            output[k].append(v)

    for k, v in six.iteritems(example):
        output[k] = np.array(output[k])
    return output
