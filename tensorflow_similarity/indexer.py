"Index embedding to allow distance based lookup"

import json
from time import time
import numpy as np
from collections import defaultdict
from collections import deque
from tabulate import tabulate
from pathlib import Path
import tensorflow as tf
from tqdm.auto import tqdm

from .matchers import NMSLibMatcher
from .tables import MemoryTable
from .evaluators import MemoryEvaluator


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
                 evaluator="memory",
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
        self.evaluator_type = evaluator

        # stats configuration
        self.stat_buffer_size = stat_buffer_size

        # calibrations scores
        self.calibration = None

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

        # mapper from id to record data
        if self.table_type == 'memory':
            self.table = MemoryTable()
        else:
            raise ValueError("Unknown table type")

        # index score normalizer
        if self.evaluator_type == 'memory':
            self.evaluator = MemoryEvaluator()
        else:
            raise ValueError("Unknown scorer type")

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
        self.matcher.add(embedding, idx, build=build, verbose=verbose)

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
            print('|-Storing data points in index table')
        idxs = self.table.batch_add(embeddings, labels, data)

        self.matcher.batch_add(embeddings, idxs, build=build, verbose=verbose)

    def single_lookup(self, embedding, k=5, as_dict=True):
        """Find the k closest match of a given embedding

        Args:
            embedding ([type]): [description]
            k (int, optional): [description]. Defaults to 5.
            as_dict(bool): return data as a dictionary. If False return as
            list(lists). Default to True.
        Returns
            if as_dict:
                list(dicts): [{"embedding", "distance", "label", "data"}]
            else:
                list(lists): embeddings, distances, labels, data
        """
        start = time()
        idxs, distances = self.matcher.lookup(embedding, k=k)
        embeddings, labels, data = self.table.batch_get(idxs)

        lookup_time = time() - start
        self._lookup_timings_buffer.append(lookup_time)
        self._stats['num_lookups'] += 1

        if as_dict:
            results = []
            for i in range(len(embeddings)):
                results.append({
                    "embedding": embeddings[i],
                    "distance": distances[i],
                    "label": labels[i],
                    "data": data[i]
                })
            return results
        else:
            return embeddings, distances, labels, data

    def _batch_lookup(self, embeddings, k=5, threads=4):
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

    # evaluation related functionm

    def is_calibrated(self):
        "Is the index calibrated?"
        return self.evaluator.is_calibrated()

    def calibrate(self, embeddings, y, k, targets, verbose=1):
        # FIXME: batch_lookup
        y_true = []
        labels = []
        distances = []

        num_examples = len(y)

        # getting the NN distance and labels
        if verbose:
            pb = tqdm(total=num_examples, desc='Finding NN')

        for idx in range(num_examples):

            # FIXME use bulk_lookup
            knn = self.single_lookup(embeddings[idx], k=k)

            for nn in knn:
                # !don't remove the casts or later code will be very slow.
                distances.append(float(nn['distance']))
                y_true.append(int(y[idx]))
                labels.append(int(nn['label']))

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        return self.evaluator.calibrate(y_true,
                                        labels,
                                        distances,
                                        targets,
                                        verbose=verbose)

    def match(self, embeddings, no_match_label=-1, verbose=1):
        """Match embeddings against the various cutpoints thresholds

        Args:
            embeddings (tensors): embeddings

            labels(list(int)): the labels associated with the embeddings.

            no_match_label (int, optional): What label value to assign when
            there is no match. Defaults to -1.

            verbose (int): display progression. Default to 1.

        Notes:
            1. its up to the model code to decide which of the cutpoints to
            use / show to the users. The evaluator returns all of them as there
            are little performance downside and makes code clearer and simpler.

            2. the function is responsible to return the list of class matched
            to allows implementation to use additional criterias if they choose
            to.

        Returns:
            dict:{cutpoint: list(bool)}
        """
        if verbose:
            pb = tqdm(total=len(embeddings), desc='looking up embeddings')

        distances = []
        labels = []
        for idx in range(len(embeddings)):

            # FIXME batch lookup
            knn = self.single_lookup(embeddings[idx], k=1)

            # ! don't remove casts, otherwise eval get very slow.
            distances.append(float(knn[0]['distance']))
            labels.append(int(knn[0]['label']))

            if verbose:
                pb.update()

        if verbose:
            pb.close()

        return self.evaluator.match(distances,
                                    labels,
                                    no_match_label=no_match_label,
                                    verbose=verbose)

    def save(self, path, compression=True):
        """Save the index to disk

        Args:
            path (str): directory where to save the index
            compression (bool, optional): Store index data compressed.
            Defaults to True.
        """
        path = str(path)
        # saving metadata
        metadata = {
            "distance": self.distance,
            "table": self.table_type,
            "evaluator": self.evaluator_type,
            "match_algorithm": self.match_algorithm,
            "stat_buffer_size": self.stat_buffer_size,
            "calibration": self.calibration
        }

        metadata['evaluator_config'] = self.evaluator.to_config()
        metadata_fname = self.__make_metadata_fname(path)
        tf.io.write_file(metadata_fname, json.dumps(metadata))

        self.table.save(path, compression=compression)
        self.matcher.save(path)

    @staticmethod
    def load(path, verbose=1):
        path = str(path)
        # recreate the index from metadata
        metadata_fname = Indexer.__make_metadata_fname(path)
        metadata = tf.io.read_file(metadata_fname)
        metadata = tf.keras.backend.eval(metadata)
        md = json.loads(metadata)
        index = Indexer(distance=md['distance'],
                        table=md['table'],
                        evaluator=md['evaluator'],
                        match_algorithm=md['match_algorithm'],
                        stat_buffer_size=md['stat_buffer_size'])

        # reload evaluator calibration if they exist
        if md['evaluator_config']:
            print("Loading evaluator calibration data")
            index.evaluator = index.evaluator.from_config(
                md['evaluator_config'])

        # reload the tables
        if verbose:
            print("Loading index data")
        index.table.load(path)

        # rebuild the index
        if verbose:
            print('Loading index matcher')
        index.matcher.load(path)
        return index

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
        # compute statistics
        stats = self.stats()

        # info
        print('[Info]')
        rows = [
            ['distance', self.distance],
            ['index table', self.table_type],
            ['matching algorithm', self.match_algorithm],
            ['evaluator', self.evaluator_type],
            ['index size', self.size()]
        ]
        print(tabulate(rows))
        print('\n')

        print('\n[Performance]')
        rows = [['num lookups', stats['num_lookups']]]
        for k, v in stats['query_performance'].items():
            rows.append([k, v])
        print(tabulate(rows))

    @staticmethod
    def __make_metadata_fname(path):
        return str(Path(path) / 'index_metadata.json')
