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

import math
import base64
import collections
import json
import tensorflow_similarity
from tensorflow_similarity.model import Moirai
from tensorflow_similarity.utils.config_utils import deserialize_moirai_object
from tensorflow_similarity.utils.config_utils import serialize_moirai_object
from tensorflow_similarity.utils.config_utils import json_dict_to_moirai_obj
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow_similarity.api.engine.nearest_neighbors import get_best_available_nearest_neighbor_algorithm
from tensorflow_similarity.api.engine.inference import InferenceRequest, Inference

from tensorflow_similarity.bundle.tensorflow_utils import load_graph_from_buffer
import six
import tensorflow as tf
from tensorflow.python.client import session
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python import Session, GraphDef

# This doesn't belong here - maybe in the UI.


def boolify(v, default_value=False):
    if not v:
        return default_value
    return (v is True or v == "True" or v == "true" or v == "t" or v == "T")


class BundleInference(Inference):
    def __init__(self, bundle):
        self.bundle = bundle

        super(BundleInference, self).__init__(
            points=bundle.get_points(),
            labels=bundle.get_labels(),
            metadata=bundle.get_metadata(),
            global_thresholds=bundle.get_global_thresholds())

        self.session = Session()
        with self.session.as_default() as sess:
            with self.session.graph.as_default() as graph:
                self.input_names, self.output_name = load_graph_from_buffer(
                    graph, bundle.get_serialized_model(),
                    bundle.get_input_nodes(), bundle.get_output_node())

    def required_inputs(self):
        features = set()
        for output_feature_name in self.dataset_config.ordered_output_features:
            transform = self.dataset_config.transforms[output_feature_name]
            input_feature_name = transform.input_feature
            features.add(input_feature_name)

        return sorted(list(features))

    def processed_input_names(self):
        return sorted(list(self.dataset_config.ordered_output_features))

    def _compute_embeddings(self, examples, request):
        input_dict = {}

        for feature_name, feature_value in examples.items():
            tensor_name = self.input_names.get(feature_name, None)
            if tensor_name:
                input_dict[tensor_name] = feature_value

        with self.session.as_default() as sess:
            with sess.graph.as_default() as graph:
                print("\n\n")
                print(input_dict)
                o = sess.run(self.output_name, feed_dict=input_dict)
                return o

    def compute_embeddings(self, examples, request):
        output = []
        one_key = [x for x in examples.keys()][0]
        num_examples = len(examples[one_key])

        print("Examples")
        print(examples)

        example_batches = []
        for start in range(0, num_examples, request.batch_size):
            input_dict = {}
            for k, v in six.iteritems(examples):
                end = start + request.batch_size
                end = min(end, num_examples)
                input_dict[k] = v[start:end]
            example_batches.append(input_dict)

        pbar = tqdm(total=len(examples), desc="Computing embeddings",
                    disable=not request.progress_bar)

        for batch in example_batches:
            try:
                output.extend(self._compute_embeddings(batch, request))
            except BaseException:
                print("Failed to process batch:")
                import traceback
                traceback.print_exc()
            pbar.update(request.batch_size)
        pbar.close()
        return output


def decode_json_blob(blob):
    blob = base64.b64decode(blob)
    blob = blob.decode("utf-8")
    return json.loads(blob)


def write_json_blob(writer, item):
    print(item)
    item = json.dumps(item)
    item = item.encode("utf-8")
    item = base64.b64encode(item)
    writer.write(item)


class BundleWrapper(object):
    def __init__(self,
                 model,
                 results,
                 labels,
                 metadata,
                 global_thresholds=[],
                 input_nodes=[],
                 output_node=None):
        self.model = model
        self.results = results
        self.labels = labels
        self.metadata = metadata
        self.global_thresholds = []
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.set_global_thresholds(global_thresholds)

    def get_inference(self):
        return BundleInference(self)

    def get_serialized_model(self):
        return self.model

    def get_graphdef(self):
        g = graph_pb2.GraphDef()
        g.ParseFromString(self.get_serialized_model())
        return g

    def get_points(self):
        return self.results

    def get_labels(self):
        return self.labels

    def get_metadata(self):
        return self.metadata

    def get_input_nodes(self):
        return self.input_nodes

    def get_output_node(self):
        return self.output_node

    def set_global_thresholds(self, t):
        self.global_thresholds = [(threshold, precision)
                                  for threshold, precision in t]

    def get_global_thresholds(self):
        return self.global_thresholds

    def _create_bundle_metadata(self):
        metadata = {
            "moirai_bundle_format": 2,
            "input_nodes": self.input_nodes,
            "output_node": self.output_node,
            "global_thresholds": self.global_thresholds,
        }
        return metadata

    def write(self, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            write_json_blob(writer, self._create_bundle_metadata())
            writer.write(self.get_serialized_model())
            write_json_blob(writer, self.get_points())
            write_json_blob(writer, [x for x in self.get_labels()])
            write_json_blob(writer, [x for x in self.get_metadata()])

    def dump(self, verbose):
        if verbose:
            print(self._create_bundle_metadata())
            print(self.get_points())
            print(self.get_labels())
            print(self.get_metadata())
        else:
            print(self._create_bundle_metadata())
            print(self.get_points()[:10], "...")
            print(self.get_labels()[:10], "...")
            print(self.get_metadata())

    @classmethod
    def load(cls, filename):
        try:
            reader = tf.io.tf_record_iterator(path=filename)
        except BaseException:
            reader = tf.compat.v1.io.tf_record_iterator(path=filename)

        records = [x for x in reader]

        (bundle_metadata, serialized_model, points, labels, metadata) = records

        bundle_metadata = decode_json_blob(bundle_metadata)
        metadata = decode_json_blob(metadata)
        points = decode_json_blob(points)
        labels = decode_json_blob(labels)
        global_thresholds = bundle_metadata.get('global_thresholds', [])

        return BundleWrapper(
            serialized_model,
            points,
            labels,
            metadata,
            global_thresholds=global_thresholds,
            input_nodes=bundle_metadata['input_nodes'],
            output_node=bundle_metadata['output_node'])
