"Index embedding to allow distance based lookup"

import nmslib
from time import time
import numpy as np
from collections import defaultdict
from collections import deque
from tabulate import tabulate
from tqdm.auto import tqdm

from .mappers import MemoryMapper
from .metrics import cosine
from operator import itemgetter
import tensorflow as tf


class Indexer():

    def __init__(self,
                 distance='cosine',
                 mapper='memory',
                 matcher='hnsw',
                 stat_buffer_size=100):

        if distance == 'cosine':
            self.space_name = 'cosinesimil'
        else:
            raise ValueError('Unsupported metric space')

        # internal structure naming
        self.matcher_method = matcher
        self.mapper_type = mapper

        # stats configuration
        self.stat_buffer_size = stat_buffer_size

        # initialize internal structures
        self._init_structures()

    def reset(self):
        "Reinitialize the indexer"
        self._init_structures()

    def _init_structures(self):
        "(re)intialize internal storage structure"

        self.matcher = nmslib.init(method=self.matcher_method,
                                   space=self.space_name)

        # mapper id > data
        if self.mapper_type == 'memory':
            self.mapper = MemoryMapper()
        else:
            self.mapper = self.mapper_type

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
        idx = self._store_data(embedding, label, data)

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
        batch = []
        batch_idxs = []

        if verbose:
            pb = tqdm(total=len(embeddings),
                      desc='Storing embeddings',
                      unit='embedings')

        for idx in range(len(embeddings)):
            emb = embeddings[idx]
            lbl = labels[idx] if not isinstance(labels, type(None)) else None
            dta = data[idx] if not isinstance(data, type(None)) else None

            # store the point
            idx = self._store_data(emb, lbl, dta)

            # build batch
            batch.append(emb)
            batch_idxs.append(idx)

            if verbose:
                pb.update(1)

        # add point to the index
        self.matcher.addDataPointBatch(batch, batch_idxs)

        if verbose:
            pb.close()

        if build:
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
            list: list of k nearest matched embeddings.
        """

        results = []
        start = time()
        idxs, distances = self.matcher.knnQuery(embedding, k=k)
        for i, idx in enumerate(idxs):
            data = self.mapper.get(idx)
            data['distance'] = distances[i]
            results.append(data)
        lookup_time = time() - start
        self._lookup_timings_buffer.append(lookup_time)
        self._stats['num_lookups'] += 1

        return results

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
        results = []
        start = time()
        matches = self.matcher.knnQueryBatch(embeddings,
                                             k=k,
                                             num_threads=threads)
        for emb_idx, res in enumerate(matches):
            elt_idxs, _ = res

            ngbs = []
            for i, e_idx in enumerate(elt_idxs):
                data = self.mapper.get(e_idx)
                ngbs.append(data)

            ngb_embs = tf.constant([n['embedding'] for n in ngbs])
            #print(ngb_embs.shape)
            emb = tf.expand_dims(embeddings[emb_idx], axis=0)
            #print(emb.shape)
            distances = cosine(emb, ngb_embs)[0]


            for idx in range(k):
                #FIXME numerical stability
                ngbs[idx]['distance'] = float(distances[idx])

            # ngbs = sorted(ngbs, key=itemgetter('distance'))

            results.append(ngbs)

        lookup_time = time() - start

        # stats
        elt_lookup_time = lookup_time / len(results)
        for _ in range(len(results)):
            self._lookup_timings_buffer.append(elt_lookup_time)
        self._stats['num_lookups'] += len(results)

        return results

    def save(self):
        raise NotImplementedError('WIP')
        info = {
            'distance': self.distance,
            'space_name': self.space_name,
            'batch_size': self.batch_size,
            'metadata': self.metadata
        }

        # serialize index
        # serialize mapping

        return info

    def load(self):
        raise NotImplementedError('WIP')

    def _build_result(target_embbing, neighboors):
        pass

    def _store_data(self, embedding, label, data):
        "store data using mapper and assign it an id"
        data = {"embedding": embedding, 'label': label, 'data': data}
        return self.mapper.add(data)

    def stats(self):
        """return index statistics"""
        stats = self._stats
        stats['num_items'] = self.mapper.size()
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
