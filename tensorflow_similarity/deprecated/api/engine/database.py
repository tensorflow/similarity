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

import numpy as np
import collections
from tensorflow_similarity.api.engine.nearest_neighbors import get_available_algorithms
from tensorflow_similarity.api.engine.nearest_neighbors import get_best_available_nearest_neighbor_algorithm  # nopep8
from tensorflow_similarity.utils.config_utils import register_custom_object

LabeledNeighbor = collections.namedtuple("Neighbor",
                                         ["distance", "index", "label"])


def get_nearest_neighbors_class(neighbor_strategy=None):
    if neighbor_strategy:
        nn_algos = get_available_algorithms()
        assert neighbor_strategy in nn_algos.keys(
        ), "Unknown nearest neighbor algorithm."
        return nn_algos[neighbor_strategy]
    else:
        return get_best_available_nearest_neighbor_algorithm()


class Database(object):
    """Database encapsulates a set of labeled points on which we know we're
    going to want to make nearest-neighbors lookups.

    Example:

        >>> import numpy as np
        >>> x = np.array([[1,2,3], [2,3,4]])
        >>> y = np.array([42, 137])
        >>> database = Database(x, y, neighbor_strategy='balltree')
        >>> neighbors_of_1_2_1 = database.query(np.array([[1,2,1]]), N=2)[0]
        >>> assert neighbors_of_1_2_1[0].index == 0
        >>> assert neighbors_of_1_2_1[0].label == 42
    """

    def __init__(self, embeddings, labels, neighbor_strategy=None):
        if len(embeddings) != len(labels):
            raise ValueError(
                "Shape of embeddings does not match labels - (%s) vs (%s)" % (
                    np.shape(embeddings), np.shape(labels)
                )
            )

        self.embeddings = embeddings
        self.labels = labels
        self.nn_cls = get_nearest_neighbors_class(neighbor_strategy)
        self.nn = self.nn_cls(embeddings)
        self.neighbor_strategy = neighbor_strategy

    def get_config(self):
        embeddings = self.embeddings
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        labels = self.labels
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        return {
            "embeddings": embeddings,
            "labels": labels,
            "neighbor_strategy": self.neighbor_strategy
        }

    def query(self, x, N=None):
        if not N:
            N = len(self.embeddings)

        assert N <= len(self.embeddings), (
            "Cannot query N=%d neighbors when we only have %d points." % (
                N, len(self.embeddings)))

        neighbors_by_point = self.nn.query(x, k=N)

        output = []
        for point_neighbors in neighbors_by_point:
            point_output = []
            for neighbor in point_neighbors:

                point_output.append(
                    LabeledNeighbor(
                        index=neighbor.index,
                        distance=neighbor.distance,
                        label=self.labels[neighbor.index]))
            output.append(point_output)
        return output

    def get_embeddings(self):
        return self.embeddings

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    def get_labels(self):
        return self.labels

    def set_labels(self, labels):
        self.labels = labels

    def get_neighbor_strategy(self):
        return self.neighbor_strategy


register_custom_object("Database", Database)
