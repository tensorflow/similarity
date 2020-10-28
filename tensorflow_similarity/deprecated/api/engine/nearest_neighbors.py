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

from collections import namedtuple
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
import sys
import pkgutil
import collections

# Neighbor has the distance from the queried point to the item, and the index
# of that item within the data points that were provided.
Neighbor = collections.namedtuple("Neighbor", ["distance", "index"])


class BaseNearestNeighbors(object):
    def __init__(self, dataset):
        pass

    def query_one(self, one, k=1):
        """
        Returns:
        A "neighbors" pair - that is, a pair where the first
        element is a list of distances, and the second element
        is a parallel list of indices.
        """
        pass

    def query(self, many, k=1):
        """Query the k nearest neighbors for many points.

        Returns:
          An iterable where the Nth element is a "neighbors"
          tuple, as above.
        """

        return [self.query_one(x, k=k) for x in many]


__AVAILABLE_NNS = {}

if pkgutil.find_loader("annoy"):
    from annoy import AnnoyIndex

    class AnnoyNeighbors(BaseNearestNeighbors):
        def __init__(self, dataset):
            self.index = AnnoyIndex(len(dataset[0]))
            for i, item in enumerate(dataset):
                self.index.add_item(i, item)
            self.index.build(len(dataset[0]) * 2)

        def query_one(self, one, k=1):
            indices, distances = self.index.get_nns_by_vector(
                one, k, include_distances=True)
            return [
                Neighbor(distance=x[0], index=x[1])
                for x in zip(distances, indices)
            ]

        def query(self, many, k=1):
            output = []
            for one in many:
                output.append(self.query_one(one, k=k))
            return output


# In practice, this algorithm doesn't seem to give decent results.
# Unclear if I've hit a bug, or if I'm misusing it.
#    __AVAILABLE_NNS['annoy'] = AnnoyNeighbors

if pkgutil.find_loader("nmslib"):
    import nmslib

    class VPTreeNeighbors(BaseNearestNeighbors):
        def __init__(self, dataset):
            space_type = 'l2'
            method = 'vptree'
            self.index = nmslib.init(space=space_type, method=method)
            for i, item in enumerate(dataset):
                self.index.addDataPoint(i, item)
            self.index.createIndex()

        def query_one(self, one, k=1):
            indices, distances = self.index.knnQuery(one, k=k)
            return [
                Neighbor(distance=x[0], index=x[1])
                for x in zip(distances, indices)
            ]

        def query(self, many, k=1):
            # Results are an iterable of neighbors, consisting of a pair with an
            # array of distances and an array of indices.
            neighbors = self.index.knnQueryBatch(many, k=k, num_threads=12)
            output = []
            for neighbor in neighbors:
                indices = neighbor[0]
                distances = neighbor[1].tolist()
                output.append([
                    Neighbor(distance=x[0], index=x[1])
                    for x in zip(distances, indices)
                ])
            return output

    __AVAILABLE_NNS['vptree'] = VPTreeNeighbors

if pkgutil.find_loader("sklearn"):
    from sklearn.neighbors import BallTree

    class BallTreeNearestNeighbors(BaseNearestNeighbors):
        def __init__(self, dataset):
            self.index = BallTree(dataset)

        def query_one(self, one, k=1):
            distances, indices = self.index.query(one.reshape(1, -1), k=k)

            distances = distances[0]
            indices = indices[0]

            output = []
            for d, i in zip(distances, indices):
                output.append(Neighbor(distance=d, index=i))
            return output

        def query(self, many, k=1):
            """The results of a query on many elements is a pair of lists
            (distances, indices), where the Nth entry in the list is a pair
            or lists:
            (distances_from_element_n, indices)

            We want a list where the Nth element is

            (distance_from_el_n, index), (distance_from_el_n, index)....
            """

            distances, indices = self.index.query(many, k=k)
            output = []
            for el_distances, el_indices in zip(distances, indices):
                output.append([
                    Neighbor(distance=x[0], index=x[1])
                    for x in zip(el_distances, el_indices)
                ])
            return output

    __AVAILABLE_NNS['balltree'] = BallTreeNearestNeighbors

if pkgutil.find_loader("falconn"):
    import falconn

    class FalcoNN(BaseNearestNeighbors):
        def __init__(self, dataset):
            self.query_object = self.__construct_index(dataset)
            self.dataset = dataset

        def __construct_index(self, dataset):
            params_hp = falconn.LSHConstructionParameters()
            params_hp.dimension = 128
            params_hp.lsh_family = falconn.LSHFamily.Hyperplane
            params_hp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
            params_hp.storage_hash_table = falconn.StorageHashTable.FlatHashTable
            params_hp.k = 19
            params_hp.l = 10
            params_hp.num_setup_threads = 0
            params_hp.seed = 37 ^ 833840234
            hp_table = falconn.LSHIndex(params_hp)
            hp_table.setup(dataset)
            qo = hp_table.construct_query_object()
            qo.set_num_probes(2464)
            return qo

        def query_one(self, one, k=1):
            indices = self.query_object.find_k_nearest_neighbors(one, k)
            pts = np.take(self.dataset, indices)
            distances = []
            for pt in pts:
                distances.append(np.linalg.norm(one - pt, ord=2).asscalar())
            return [
                Neighbor(distance=x[0], index=x[1])
                for x in zip(distances, indices)
            ]


#    __AVAILABLE_NNS['falconn'] = FalcoNN


class SlowNearestNeighbors(BaseNearestNeighbors):
    def __init__(self, dataset):
        print("Install 'sklearn', 'annoy', or 'nmslib' to avoid using the "
              "extremely slow numpy implementation.")
        self.dataset = np.asarray(dataset)

    def query(self, many, k=1):
        return [self.query_one(one, k=k) for one in many]

    def query_one(self, one, k=1):
        one = np.asarray(one)
        d = self.dataset - one
        d = d * d
        d = np.sum(d, axis=1)

        indices = [x for x in range(len(self.dataset))]
        indices.sort(key=lambda x: d[x])
        indices = indices[:k]
        distances = np.take(d, indices)
        return [
            Neighbor(distance=x[0], index=x[1])
            for x in zip(distances, indices)
        ]


__AVAILABLE_NNS['slow'] = SlowNearestNeighbors


def get_available_algorithms():
    return __AVAILABLE_NNS


def get_best_available_nearest_neighbor_algorithm():
    ordered = ["vptree", "balltree", "slow"]
    for algorithm in ordered:
        if algorithm in __AVAILABLE_NNS:
            return __AVAILABLE_NNS[algorithm]

    raise ValueError("No available nearest neighbor algorithms.")


if __name__ == '__main__':
    import sys
    target_set = np.random.normal(size=(10000, 128))
    test_set = np.random.normal(size=(1000, 128))

    test = test_set[0:1]
    test2 = test_set[0]

    sets = collections.defaultdict(set)

    for name, cls in __AVAILABLE_NNS.items():
        print(name)
        nn = cls(target_set)
        result = nn.query_one(test2, k=10)
        for d, i in result:
            print("  %f %s" % (d, i))
            sets[name].add(i)

        print(" Batch")
        results = nn.query(test, k=10)
        for result in results:
            for d, i in result:
                print("  %s %s" % (d, i))
                sets[name + "_batch"].add(i)

    for k, v in sets.items():
        for k2, v2 in sets.items():
            vs = set(v)
            v2s = set(v2)

            print("%s/%s - %f" % (k, k2, 1 - float(len(vs - v2s)) / len(vs)))
