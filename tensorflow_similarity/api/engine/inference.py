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

import bisect
import collections
import json
from tensorflow_similarity.utils.config_utils import serialize_moirai_object
from tensorflow_similarity.api.engine.nearest_neighbors import get_best_available_nearest_neighbor_algorithm
from tensorflow_similarity.dataset import Dataset, DatasetConfig
import numpy as np
import six


def boolify(v, default_value=False):
    if not v:
        return default_value
    return (v or v == "True" or v == "true" or v == "t" or v == "T")


class InferenceRequest(object):
    def __init__(self,
                 num_neighbors=1,
                 include_examples=False,
                 include_input_embeddings=False,
                 preprocess=True,
                 augment=False,
                 include_target_embeddings=False,
                 include_target_metadata=False,
                 progress_bar=False,
                 batch_size=32,
                 **kwargs):
        # Number of neighbors to retrieve for nearest neighbor lookups
        self.num_neighbors = int(num_neighbors)

        # Responses should include the (raw) examples provided.
        self.include_examples = boolify(include_examples)

        # Response should include the embedding for the input(s)
        self.include_input_embeddings = boolify(include_input_embeddings)

        self.include_target_metadata = boolify(include_target_metadata)

        # Run the configured preprocessing for the input
        self.preprocess = boolify(preprocess, default_value=True)

        # Run the configured preprocessing for the input
        self.augment = boolify(augment, default_value=False)

        # For nearest neighbors, return the embeddings as well
        # as the label / metadata.
        self.include_target_embeddings = boolify(include_target_embeddings)

        # For local clients, set to true to show a progress bar.
        self.progress_bar = progress_bar

        self.batch_size = batch_size


class Inference(object):
    def __init__(self,
                 points=[],
                 labels=[],
                 metadata=[],
                 global_thresholds=None):
        self.points = points
        self.labels = labels
        self.metadata = metadata
        self.num_points = len(points)
        nn = get_best_available_nearest_neighbor_algorithm()
        self.tree = nn(np.array(self.points))
        self.global_thresholds = global_thresholds

    def is_calibrated(self):
        return len(self.global_thresholds) > 0

    def num_classes(self):
        return len(self.labels)

    def preprocess(self, examples, request=InferenceRequest()):
        """Raw example is either:
        A dictionary of lists of feature values, where the Nth entry in each of
        the lists comes from the Nth data point OR
        A list of dictionaries, where each dictionary represents the data for
        a single element.
        """

        if isinstance(examples, list):
            tmp = collections.defaultdict(list)
            for example in examples:
                for k, v in six.iteritems(example):
                    tmp[k].append(v)
            return tmp
        else:
            return examples

    def compute_embeddings(self, examples, request):
        raise NotImplementedError

    def embed(self, raw_examples, request=InferenceRequest()):
        examples = self.preprocess(raw_examples, request)
        embeddings = self.compute_embeddings(examples, request)

        response = []
        for raw_example, embedding in zip(raw_examples, embeddings):
            element = {}
            element['embedding'] = embedding.tolist()
            if request.include_examples:
                element['example'] = raw_example
            response.append(element)
        return response

    def get_precision(self, distance, label):
        if len(self.global_thresholds):
            thresholds = self.global_thresholds
        else:
            # No threshold data provided.
            return None

        idx = bisect.bisect_right(thresholds, (distance, 1.0))
        if idx > 0:
            idx -= 1
        if idx < len(thresholds):
            lo = thresholds[idx]
            hi = thresholds[idx + 1]

            r = hi[0] - lo[0]
            v = distance - lo[0]
            base_precision = lo[1]
            delta_precision = hi[1] - lo[1]
            delta_fraction = (float(v) / r)

            interpolated_precision = (
                delta_fraction * delta_precision + base_precision)
            return interpolated_precision
        else:
            return thresholds[idx]

    def neighbors(self, raw_examples, request=InferenceRequest()):
        response = []
        embed_response = self.embed(raw_examples, request=request)
        input_embeddings = np.array([e['embedding'] for e in embed_response])

        if len(input_embeddings) == 1:
            nn_results = self.tree.query_one(input_embeddings,
                                             k=request.num_neighbors)

            nn_results = [[x for x in nn_results]]
        else:
            nn_results = self.tree.query(
                input_embeddings, k=request.num_neighbors)

            nn_results = [[x for x in r] for r in nn_results]

        for raw_example, example_embedding, results in zip(
                raw_examples, input_embeddings, nn_results):
            element_response = collections.defaultdict(list)
            if request.include_examples:
                element_response['example'] = raw_example

            if request.include_input_embeddings:
                element_response['embedding'] = example_embedding

            for distance, idx in results:
                element_response['labels'].append(self.labels[idx])
                element_response['distances'].append(distance)
                precision = self.get_precision(distance, idx)
                if len(self.metadata) and request.include_target_metadata:
                    metadata = self.metadata[idx]
                    try:
                        val = json.loads(metadata)
                    except BaseException:
                        val = metadata
                    element_response['metadata'].append(val)
                else:
                    element_response['metadata'].append({
                        "display_data": self.labels[idx],
                        "display_renderer": "TextRenderer"})
                if precision:
                    element_response['precisions'].append(precision)

                if request.include_target_embeddings:
                    element_response['target_embeddings'].append(
                        self.points[idx].tolist())
                    element_response['target_distance'].append(
                        np.linalg.norm(
                            self.points[idx] - example_embedding, ord=2))

            response.append(element_response)
        return response
