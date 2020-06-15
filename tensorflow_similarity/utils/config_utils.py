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
import copy
import collections
import logging
import importlib
import json
from tensorflow.keras.utils import deserialize_keras_object, get_custom_objects, serialize_keras_object
import os
import platform
import tensorflow as tf
import subprocess

flags.DEFINE_string(
    'custom_dependency_module', None,
    'Optional python module to load before loading the configuration.')

FLAGS = flags.FLAGS


def load_custom_configuration(module=None):
    print("LCC", module, FLAGS.custom_dependency_module)
    if module:
        importlib.import_module(module)
    elif FLAGS.custom_dependency_module and len(
            FLAGS.custom_dependency_module) > 0:
        importlib.import_module(FLAGS.custom_dependency_module)


def register_custom_object(name, cls):
    """Wrapper around keras's custom object dictionary."""
    get_custom_objects()[name] = cls


def load_json_configuration_file(filename):
    with tf.io.gfile.GFile(filename, "r") as f:
        config = f.read()

        pwd = os.getcwd()
        pwd = pwd.replace("\\", "\\\\")
        hostname = platform.uname()[1]

        config = config % {'pwd': pwd, 'hostname': hostname}
        config = json.loads(config)
        return config


class GlobalWrapper(object):
    global_vars = {}

    def __init__(self, key=None):
        self.key = key

    def __call__(self):
        return GlobalWrapper.global_vars.get(self.key, None)

    @classmethod
    def set_globals(cls, global_vars):
        cls.global_vars = global_vars


register_custom_object("GLOBALS", GlobalWrapper)


class ArgsWrapper(object):
    args = {}

    def __init__(self, key=None, default_value=None, type="str", help=None):
        self.key = key
        self.default_value = default_value

    def __call__(self):
        if self.key not in flags.FLAGS:
            return self.default_value

        v = flags.FLAGS[self.key]
        if v is None:
            return self.default_value
        v = v.value
        if v is None:
            return self.default_value
        return v


register_custom_object("ARGS", ArgsWrapper)


def set_globals(global_dict):
    GlobalWrapper.set_globals(global_dict)


def serialize_moirai_object(obj):
    if obj is None:
        return None
    return serialize_keras_object(obj)


def deserialize_moirai_object(obj):
    if isinstance(obj, dict) and 'class_name' in obj and 'config' in obj:
        obj = deserialize_keras_object(obj)
    elif isinstance(obj, dict) and 'type' in obj:
        obj = json_dict_to_moirai_obj(obj)

    # Custom Moirai wrappers which either refer to some global object,
    # or some command line argument. In either case, call the object to get the
    # underlying value, and then recursively deserialize that object.
    if isinstance(obj, ArgsWrapper) or isinstance(obj, GlobalWrapper):
        obj = obj()

        return deserialize_moirai_object(obj)

    if obj is None:
        return None
    return obj


def default_object_for_name(name):
    config = {
        "class_name": name,
        "config": {

        }
    }
    return deserialize_moirai_object(config)


def value_or_callable(fv):
    """If the argument is a callable, call it and return the value, otherwise
    return the value directly."""
    if callable(fv):
        return fv()
    if isinstance(fv, dict):
        fv = deserialize_keras_object(fv)
        return fv()
    else:
        return fv


def recursively_update_dict(d, updates):
    for k, v in updates.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursively_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def json_dict_to_moirai_obj(d):
    d = copy.deepcopy(d)
    if d is None:
        return None
    cls = d['type']
    del d['type']
    config = {"class_name": cls, "config": d}
    return deserialize_moirai_object(config)


def get_module_version(module):
    module_dir = os.path.dirname(module.__file__)
    cd = "cd " + module_dir + " ; git describe --always"
    print(cd)
    try:
        out = subprocess.check_output(cd, shell=True)
        out = out.decode("utf-8").strip()
        return out
    except BaseException:
        if hasattr(module, "__version__"):
            return module.__version__
