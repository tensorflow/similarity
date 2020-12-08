"Index embedding to allow distance based lookup"

import json
import nmslib
from time import time
import numpy as np
from collections import defaultdict
from collections import deque
from tabulate import tabulate
from pathlib import Path
import tensorflow as tf

from .matchers import NMSLibMatcher
from .tables import MemoryTable


class Indexer():
    # semantic sugar for the order of the returned data
    EMBEDDINGS = 0
    DISTANCES = 1
    LABELS = 2
    DATA = 3

    def __init__(self,
                 distance='cosine',
                 table='memory',
                 match_algorithm='nmslib_hnsw',
                 stat_buffer_size=1000):
        """Index embeddings to make them searchable via KNN

        Args:
            distance (str, optional): Distance type used in the embeedings.
            Defaults to 'cosine'.

            table (str, optional): How to store the index records.
            Defaults to 'memory'.

            matcher (str, optional): What algorithm to use to perfom KNN
            search. Defaults to 'hnsw'.

            stat_buffer_size (int, optional): Size of the sliding windows
            buffer used to computer index performance. Defaults to 1000.

        Raises:
            ValueError: [description]
        """
        self.distance = distance  # needed for save()/load()

        # internal structure naming
        self.match_algorithm = match_algorithm
        self.table_type = table

        # stats configuration
        self.stat_buffer_size = stat_buffer_size

        # initialize internal structures
        self._init_structures()

    def reset(self):
        "Reinitialize the indexer"
        self._init_structures()

    def _init_structures(self):
        "(re)intialize internal storage structure"

        if self.match_algorithm == 'nmslib_hnsw':
            self.matcher = NMSLibMatcher(self.distance, self.match_algorithm)
        else:
            raise ValueError('Unknown matching_algorithm')

        # mapper id > data
        if self.table_type == 'memory':
            self.table = MemoryTable()
        else:
            raise ValueError("Unknown table type")

        # stats
        self._stats = defaultdict(int)
        self._lookup_timings_buffer = deque([], maxlen=self.stat_buffer_size)

    def add(self, embedding, label=None, data=None, build=True, verbose=1):
        """ Add a single embedding to the indexer

        Args:
            embedding (tensor): TF similarity model output / embeddings.

            label (str/int, optional): Label(s) associated with the
            embedding. Defaults to None.

            data (Tensor, optional): Input data associated with the embedding.
            Defaults to None.

            build (bool, optional): Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batchs/points and build it manually once after.

            verbose (int, optional): Display progress if set to 1.
            Defaults to 1.
        """
        # store data and get its id
        idx = self.table.add(embedding, label, data)

        # add index to the embedding
        # !the order of parameters between addDataPoint and addDataPointBatch
        # !are inverted
        self.matcher.addDataPoint(idx, embedding)

        if build:
            self.build(verbose=verbose)

    def batch_add(self,
                  embeddings,
                  labels=None,
                  data=None,
                  build=True,
                  verbose=1):
        """Add a batch of embeddings to the indexer

        Args:
            embeddings (list(tensor)): TF similarity model output / embeddings.
            labels (list(str/int), optional): label(s) associated with the
            embedding. Defaults to None.
            datas (list(Tensor), optional): input data associated with the
            embedding. Defaults to None.

            build (bool, optional): Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batchs/points and build it manually once after.

            verbose (int, optional): Display progress if set to 1.
            Defaults to 1.
        """

        # store points
        if verbose:
            print('Storing')
        embeddings_idx = self.table.batch_add(embeddings, labels, data)

        # add point to the index
        if verbose:
            print('Indexing')
        self.matcher.addDataPointBatch(embeddings, embeddings_idx)

        if build:
            if verbose:
                print('Building')
            self.build(verbose=verbose)

    def build(self, verbose=1):
        """Build the index this is need to take into account the new points
        """
        show = True if verbose else False
        self.matcher.createIndex(print_progress=show)
        self._stats['query_time'] = 0
        self._stats['query'] = 0
        self.lookup_timing = deque([], maxlen=self.stat_buffer_size)

    def single_lookup(self, embedding, k=5):
        """Find the k closest match of a given embedding

        Args:
            embedding ([type]): [description]
            k (int, optional): [description]. Defaults to 5.
        Returns
            list(lists): embeddings, distances, labels, data
        """
        start = time()
        idxs, distances = self.matcher.knnQuery(embedding, k=k)
        embeddings, labels, data = self.table.batch_get(idxs)

        lookup_time = time() - start
        self._lookup_timings_buffer.append(lookup_time)
        self._stats['num_lookups'] += 1

        return embeddings, distances, labels, data

    def batch_lookup(self, embeddings, k=5, threads=4):
        """Find the k closest match of a batch of embeddings

        Args:
            embeddings ([type]): [description]
            k (int, optional): [description]. Defaults to 5.
            threads (int, optional). Defaults to 4
        Returns
            list(list): list of k nearest matched embeddings.
        """

        print('Unreliable method -- Distances are innacurate:%s' % int(time()))
        # results = []
        # start = time()
        # matches = self.matcher.knnQueryBatch(embeddings,
        #                                      k=k,
        #                                      num_threads=threads)
        # for emb_idx, res in enumerate(matches):
        #     elt_idxs, _ = res

        #     ngbs = []
        #     for i, e_idx in enumerate(elt_idxs):
        #         data = self.mapper.get(e_idx)
        #         ngbs.append(data)

        #     ngb_embs = tf.constant([n['embedding'] for n in ngbs])
        #     # print(ngb_embs.shape)
        #     emb = tf.expand_dims(embeddings[emb_idx], axis=0)
        #     # print(emb.shape)
        #     distances = cosine(emb, ngb_embs)[0]

        #     for idx in range(k):
        #         # FIXME numerical stability
        #         ngbs[idx]['distance'] = float(distances[idx])

        #     # ngbs = sorted(ngbs, key=itemgetter('distance'))

        #     results.append(ngbs)

        # lookup_time = time() - start

        # # stats
        # elt_lookup_time = lookup_time / len(results)
        # for _ in range(len(results)):
        #     self._lookup_timings_buffer.append(elt_lookup_time)
        # self._stats['num_lookups'] += len(results)

        # return results

    def save(self, path, compression=True):
        """Save the index to disk

        Args:
            path (str): directory where to save the index
            compression (bool, optional): Store index data compressed.
            Defaults to True.
        """
        # saving metadata
        metadata = {
            "distance": self.distance,
            "table": self.table_type,
            "matcher": self.matcher_method,
            "stat_buffer_size": self.stat_buffer_size
        }

        metadata_fname = self.__make_metadata_fname(path)
        tf.io.write_file(metadata_fname, json.dumps(metadata))

        self.table.save(path, compression=compression)

    def load(self, path, verbose=1):
        # recreate the index from metadata
        metadata_fname = self.__make_metadata_fname(path)
        md = json.loads(tf.io.read_file(metadata_fname))
        index = Indexer(distance=md['distance'],
                        table=md['table'],
                        match_algorithm=md['matcher'],
                        stat_buffer_size=md['stat_buffer_size'])

        # reload the table
        if verbose:
            print("Loading index data")
        index.table.load(path)

        # rebuild the index
        if verbose:
            print('Rebuilding index')


        raise NotImplementedError('WIP')

    def size(self):
        "Return the index size"
        return self.table.size()

    def stats(self):
        """return index statistics"""
        stats = self._stats
        stats['size'] = self.table.size()
        stats['stat_buffer_size'] = self.stat_buffer_size

        # query performance - make sure we don't count unused buffer
        max_idx = min(stats['num_lookups'], self.stat_buffer_size)
        lookup_timings = list(self._lookup_timings_buffer)[:max_idx]

        # ensure we never have an empty list
        lookup_timings = lookup_timings if list(lookup_timings) else [0]

        # compute stats
        stats['query_performance'] = {
            'min': np.min(lookup_timings),
            'max': np.max(lookup_timings),
            'avg': np.average(lookup_timings),
            'median': np.median(lookup_timings),
            'stddev': np.std(lookup_timings)
        }
        return stats

    def print_stats(self):
        "display statistics in terminal friendly fashion"
        stats = self.stats()

        rows = []
        for k, v in stats.items():
            if not isinstance(v, dict):
                rows.append([k, v])
        print('[Index statistics]')
        print(tabulate(rows))

        print('\n[Query performance]')
        rows = []
        for k, v in stats['query_performance'].items():
            rows.append([k, v])
        print(tabulate(rows))

    def __make_metadata_fname(path):
        return str(Path(path) / 'index_metadata.json')
