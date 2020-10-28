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

import logging
import os
import traceback
from collections import defaultdict, namedtuple

import numpy as np
import six
import tensorflow as tf
from flask import jsonify, render_template
from PIL import Image

from tensorflow_similarity.api.engine.inference import InferenceRequest
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow_similarity.dataset import Dataset
from tensorflow_similarity.preprocessing.identity import Identity
from tensorflow_similarity.ui2.fake_inference import FakeInference
from tensorflow_similarity.ui2.input_handler import (
    ImageInputHandler, StringInputHandler, from_config)
from tensorflow_similarity.ui2.renderers import *
from tensorflow_similarity.utils.config_utils import json_dict_to_moirai_obj

LoadedModel = namedtuple("LoadedModel", ["inference", "dataset_config"])
Result = namedtuple("Result", ["label", "distance", "precision", "metadata"])

MODELS = {}
DEFAULT_MODEL_NAME = None
INPUT_HANDLER_GEN_FN = None


def get_inference(name=None):
    global DEFAULT_MODEL_NAME
    global MODELS
    if name is None:
        name = DEFAULT_MODEL_NAME

    return MODELS[name].inference


def get_dataset_config(name=None):
    global DEFAULT_MODEL_NAME
    global MODELS
    if name is None:
        name = DEFAULT_MODEL_NAME

    return MODELS[name].dataset_config


def get_input_handlers():
    global INPUT_HANDLER_GEN_FN
    return INPUT_HANDLER_GEN_FN()


def get_default_modelname():
    return DEFAULT_MODEL_NAME


def handler_generator(input_config):
    """Function which generates a set of handler objects for
    the input config. Handler objects are lightweight and generated
    for each request."""

    def generate():
        handlers = {}
        for config_dict in input_config:
            name = config_dict["parameter_name"]
            handlers[name] = from_config(config_dict)
        return handlers
    return generate


def load_models(model_config):
    models = {}
    for model_identifier, config in model_config.items():
        if "use_stub" in config and config["use_stub"]:
            include_precisions = config.get("stub_precisions", True)
            imgs = config.get("include_display_img", False)
            texts = config.get("include_display_text", False)

            inference = FakeInference(
                include_precisions=include_precisions,
                include_display_img=imgs,
                include_display_text=texts)

            dataset_config = None
        else:
            bundle = BundleWrapper.load(config['filename'])
            inference = bundle.get_inference()
            dataset_config = json_dict_to_moirai_obj(config["dataset_config"])

        models[model_identifier] = LoadedModel(
            inference=inference, dataset_config=dataset_config)
    return models


def configure_server(config):
    global INPUT_HANDLER_GEN_FN
    global MODELS
    global DEFAULT_MODEL_NAME

    INPUT_HANDLER_GEN_FN = handler_generator(config["inputs"])
    MODELS = load_models(config["models"])
    DEFAULT_MODEL_NAME = config["default_model"]


def preprocess(features, dataset_config):
    dataset = Dataset(features, dataset_config)

    item = dataset.get_item(
        0, preprocess=True, augment=False)
    item = item.feature_dictionary

    out_features = {}

    for k, v in item.items():
        out_features[k] = np.array([v])

    return out_features


def perform_search(request, embedding_model, dataset_config, features):
    assert embedding_model is not None

    N = int(request.values.get('num_neighbors', "5"))
    include_input_emb = ('include_input_embeddings' in request.values)
    include_target_emb = ('include_target_embeddings' in request.values)

    pp_features = preprocess(features, dataset_config)

    req = InferenceRequest(
        num_neighbors=N,
        include_input_embeddings=include_input_emb,
        include_target_embeddings=include_target_emb,
        include_target_metadata=True)

    output = embedding_model.neighbors(pp_features, request=req)

    return output


def search_ui(request):
    inference = get_inference()
    dataset_config = get_dataset_config()
    has_precisions = inference.is_calibrated()

    if has_precisions:
        threshold = float(request.values.get("threshold", .9))
    else:
        threshold = float(request.values.get("threshold", 100))

    num_neighbors = int(request.values.get("num_neighbors", 5))
    verbose = "noverbose" not in request.values

    handler_dict = get_input_handlers()

    feature_values = defaultdict(list)
    feature_widgets = {}
    ready = True

    for k in sorted([x for x in handler_dict.keys()]):
        v = handler_dict[k]
        logging.info(
            "KV %s %s " % (k, v))
        value = v.get_value(request)
        if value is None:
            ready = False
        logging.info(value)
        feature_values[k].append(value)
        feature_widgets[k] = v.render(request)

    if not ready:
        return render_template(
            "results.html",
            features=six.iteritems(feature_widgets),
            results=None,
            matches=None,
            threshold=threshold,
            verbose=False,
            num_neighbors=num_neighbors,
            ready=ready,
            label=None)

    label = ''

    res = perform_search(request, inference, dataset_config,
                         feature_values)
    if res:
        res = res[0]

    labels = res['labels']
    distances = [round(d, 2) for d in res['distances']]
    precisions = res['precisions'] if 'precisions' in res else [-1] * len(
        distances)

    metadata = res['metadata']
    print(metadata)

    match_targets = []
    match_distances = []

    rendered_metadata = []

    if not len(metadata):
        for l in labels:
            rendered_metadata.append(l)
    else:
        for l, item in zip(labels, metadata):
            if 'display_renderer' in item and 'display_data' in item:
                renderer_class = item['display_renderer']
                renderer = json_dict_to_moirai_obj({"type": renderer_class})

                rendered_metadata.append(renderer.render(item['display_data']))
            else:
                rendered_metadata.append(l)

    results = [
        Result(
            label=l, distance=round(
                d, 3), precision=round(
                p, 3), metadata=m) for l, d, m, p in zip(
                    labels, distances, rendered_metadata, precisions)]

    if has_precisions:
        matches = [t for t in results if t.precision >= threshold]
    else:
        matches = [t for t in results if t.distance <= threshold]

    try:
        return render_template(
            "results.html",
            features=six.iteritems(feature_widgets),
            results=results,
            matches=matches,
            threshold=threshold,
            verbose=verbose,
            label=label,
            ready=ready,
            has_precisions=has_precisions,
            num_neighbors=num_neighbors)
    except BaseException:
        return traceback.format_exc()
