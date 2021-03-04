"Index embedding to allow distance based lookup"

import json
from time import time
import numpy as np
from collections import defaultdict, deque
from tabulate import tabulate
from pathlib import Path
import tensorflow as tf
from tqdm.auto import tqdm

# types
from typing import Dict, List, Union, DefaultDict, Deque, Optional
from .types import FloatTensorLike, PandasDataFrame, TensorLike

# internal
from .matchers import NMSLibMatcher
from .tables import MemoryTable
from .evaluators import MemoryEvaluator
from .metrics import EvalMetric, make_metric, F1Score


class Indexer():
    # semantic sugar for the order of the returned data
    EMBEDDINGS = 0
    DISTANCES = 1
    LABELS = 2
    DATA = 3
    RANKS = 4

    def __init__(self,
                 distance: str = 'cosine',
                 table: str = 'memory',
                 match_algorithm: str = 'nmslib_hnsw',
                 evaluator: str = 'memory',
                 embedding_output: int = None,
                 stat_buffer_size: int = 1000) -> None:
        """Index embeddings to make them searchable via KNN

        Args:
            distance (str, optional): Distance type used in the embeedings.
            Defaults to 'cosine'.

            table (str, optional): How to store the index records.
            Defaults to 'memory'.

            matcher (str, optional): What algorithm to use to perfom KNN
            search. Defaults to 'hnsw'.

            embedding_output (int, optional): Which model output head predicts
            the embbedings that should be indexed. Default to None which is for
            single output model. For multi-head model, the callee, usually the
            `SimilarityModel()` class is responsible for passing the
            correct one.

            stat_buffer_size (int, optional): Size of the sliding windows
            buffer used to computer index performance. Defaults to 1000.

        Raises:
            ValueError: Invalid matcher or table.
        """
        self.distance = distance  # needed for save()/load()
        self.embedding_output = embedding_output

        # internal structure naming
        self.match_algorithm = match_algorithm
        self.table_type = table
        self.evaluator_type = evaluator

        # stats configuration
        self.stat_buffer_size = stat_buffer_size

        # calibration
        self.is_calibrated = False
        self.calibration_metric: EvalMetric = F1Score()
        self.cutpoints: Dict[str, Dict[str, Union[int, float, str]]] = {}
        self.calibration_thresholds: Dict[str, List[Union[float, int]]] = {}

        # initialize internal structures
        self._init_structures()

    def reset(self) -> None:
        "Reinitialize the indexer"
        self._init_structures()

    def _init_structures(self) -> None:
        "(re)intialize internal storage structure"

        if self.match_algorithm == 'nmslib_hnsw':
            self.matcher = NMSLibMatcher(self.distance, self.match_algorithm)
        else:
            raise ValueError('Unknown matching_algorithm')

        # mapper from id to record data
        if self.table_type == 'memory':
            self.table: MemoryTable = MemoryTable()
        else:
            raise ValueError("Unknown table type")

        # code used to evaluate indexer performance
        if self.evaluator_type == 'memory':
            self.evaluator: MemoryEvaluator = MemoryEvaluator()
        else:
            raise ValueError("Unknown scorer type")

        # stats
        self._stats: DefaultDict[str, int] = defaultdict(int)
        self._lookup_timings_buffer: Deque = deque([], maxlen=self.stat_buffer_size)  # noqa

        # calibration data
        self.is_calibrated = False
        self.calibration_metric = F1Score()
        self.cutpoints = {}
        self.calibration_thresholds = {}

    def _get_embedding(self, prediction: FloatTensorLike) -> TensorLike:
        """Return the 1st embedding vector from a (multi-output) model
        prediction

        See: `single_lookup()`, `add()`

        Args:
            prediction (FloatTensorLike): Model prediction for single embedding.

        Returns:
            TensorLike: 1D Tensor that contains the actual embedding
        """

        # in multi-output: embedding is [output_num][0]
        if self.embedding_output:
            return prediction[self.embedding_output][0]
        return prediction[0]  # single output > return 1st element

    def _get_embeddings(self, predictions: FloatTensorLike) -> TensorLike:
        """Return the embedding vectors from a (multi-output) model prediction

        Args:
            predictions (FloatTensorLike): Model predictions.

        Returns:
            TensorLike: 2D Tensor (num_embeddings, embedding_value)
        """

        if self.embedding_output:
            return predictions[self.embedding_output]
        return predictions  # passthrough

    def _cast_label(self, label: Optional[int]) -> Optional[int]:
        if label is not None:
            label = int(label)
        return label

    def add(self,
            prediction: FloatTensorLike,
            label: Optional[int] = None,
            data: TensorLike = None,
            build: bool = True,
            verbose: int = 1):
        """ Add a single embedding to the indexer

        Args:
            prediction (TensorLike): TF similarity model prediction.

            label (Union[str, int], optional): Label(s) associated with the
            embedding. Defaults to None.

            data (TensorLike, optional): Input data associated with
            the embedding. Defaults to None.

            build (bool, optional): Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batchs/points and build it manually once after.

            verbose (int, optional): Display progress if set to 1.
            Defaults to 1.
        """

        # deal with potential multi-output
        embedding = self._get_embedding(prediction)

        # store data and get its id
        idx = self.table.add(embedding, label, data)

        # add index to the embedding
        self.matcher.add(embedding, idx, build=build, verbose=verbose)

    def batch_add(self,
                  predictions: TensorLike,
                  labels: Optional[List[Optional[int]]] = None,
                  data: Optional[TensorLike] = None,
                  build: bool = True,
                  verbose: int = 1):
        """Add a batch of embeddings to the indexer

        Args:
            predictions (Tensorlike): TF similarity model predictions.
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

        # deal with potential multi-output
        embeddings = self._get_embeddings(predictions)

        # store points
        if verbose:
            print('|-Storing data points in index table')
        idxs = self.table.batch_add(embeddings, labels, data)
        self.matcher.batch_add(embeddings, idxs, build=build, verbose=verbose)

    def single_lookup(self,
                      prediction: FloatTensorLike,
                      k: int = 5):
        """Find the k closest match of a given embedding

        Args:
            prediction (FloatTensorLike): model prediction.
            k (int, optional): Number of nearest neighboors to lookup.
            Defaults to 5.
        Returns
            list(dicts): [{"embedding", "distance", "label", "data"}]

        """

        embedding = self._get_embedding(prediction)
        start = time()
        idxs, distances = self.matcher.lookup(embedding, k=k)
        nn_embeddings, labels, data = self.table.batch_get(idxs)

        lookup_time = time() - start
        results = []
        for i in range(len(nn_embeddings)):
            # ! casting is needed to avoid slowness down the line
            results.append({
                "rank": i + 1,
                "embedding": nn_embeddings[i],
                "distance": float(distances[i]),
                "label": self._cast_label(labels[i]),
                "data": data[i]
            })
        self._lookup_timings_buffer.append(lookup_time)
        self._stats['num_lookups'] += 1
        return results

    def batch_lookup(self,
                     predictions: FloatTensorLike,
                     k: int = 5,
                     threads: int = None,
                     verbose: int = 1):

        embeddings = self._get_embeddings(predictions)
        num_embeddings = len(embeddings)
        start = time()

        if verbose:
            pb = tqdm(total=num_embeddings, desc='Finding NN')

        lookups = []
        for eidx in range(num_embeddings):
            # FIXME: threading / or get nsmw batch to work reliablys
            lidxs, distances = self.matcher.lookup(embeddings[eidx], k=k)
            nn_embeddings, labels, data = self.table.batch_get(lidxs)

            lookup = []
            for i in range(len(nn_embeddings)):
                lookup.append({
                    "rank": i + 1,
                    "embedding": nn_embeddings[i],
                    "distance": float(distances[i]),
                    "label": self._cast_label(labels[i]),
                    "data": data[i]
                })
            lookups.append(lookup)
            if verbose:
                pb.update()
        if verbose:
            pb.close()

        # stats
        lookup_time = time() - start
        per_lookup_time = lookup_time / num_embeddings
        for _ in range(num_embeddings):
            self._lookup_timings_buffer.append(per_lookup_time)
        self._stats['num_lookups'] += num_embeddings

        return lookups

    # evaluation related functions
    def evaluate(self, predictions, y, metrics, k: int = 1, verbose: int = 1):
        # Find NN
        lookups = self.batch_lookup(predictions, verbose=verbose)

        # Evaluate them
        return self.evaluator.evaluate(index_size=self.size(),
                                       metrics=metrics,
                                       targets_labels=y,
                                       lookups=lookups)

    def calibrate(self,
                  predictions: FloatTensorLike,
                  y: List[int],
                  thresholds_targets: Dict[str, float],
                  k: int = 1,
                  calibration_metric: Union[str, EvalMetric] = "f1_score",
                  extra_metrics: List[Union[str, EvalMetric]] = ['precision', 'recall'],  # noqa
                  rounding: int = 2,
                  verbose: int = 1):

        # find NN
        lookups = self.batch_lookup(predictions, verbose=verbose)

        # making sure our metrics are all EvalMetric object
        calibration_metric = make_metric(calibration_metric)
        # This aweful syntax is due to mypy not understanding subtype :(
        extra_eval_metrics: List[Union[str, EvalMetric]] = [make_metric(m) for m in extra_metrics]  # noqa
        # running calibration
        thresholds, cutpoints = self.evaluator.calibrate(
            self.size(),
            calibration_metric=calibration_metric,
            thresholds_targets=thresholds_targets,
            targets_labels=y,
            lookups=lookups,
            extra_metrics=extra_eval_metrics,
            metric_rounding=rounding,
            verbose=verbose
        )

        # display cutpoint results if requested
        if verbose:
            headers = ['name', 'value', 'distance']  # noqa

            # dynamicaly find which metrics we need
            for data in cutpoints.values():
                for k in data.keys():
                    if k not in headers:
                        headers.append(str(k))
                break

            # print(cutpoints)
            rows = []
            for data in cutpoints.values():
                rows.append([data[v] for v in headers])
            print(tabulate(rows, headers=headers))

        # store info for serialization purpose
        self.is_calibrated = True
        self.calibration_metric = calibration_metric
        self.cutpoints = cutpoints
        self.calibration_thresholds = thresholds
        return {"cutpoints": cutpoints, "thresholds": thresholds}

    def match(self,
              predictions: FloatTensorLike,
              no_match_label: int = -1,
              verbose: int = 1):
        """Match embeddings against the various cutpoints thresholds

        Args:
            predictions (FloatTensorLike): embeddings

            no_match_label (int, optional): What label value to assign when
            there is no match. Defaults to -1.

            verbose (int): display progression. Default to 1.

        Notes:
            1. its up to the `Model.match()` code to decide which of
            cutpoints results to use / show to the users.
            This function returns all of them as there is little performance
            downside to do so and it makes code clearer and simpler.

            2. the function is responsible to return the list of class matched
            to allows implementation to use additional criterias if they choose
            to.

        Returns:
            dict:{cutpoint: list(bool)}
        """
        lookups = self.batch_lookup(predictions, k=1, verbose=verbose)

        # vectorize
        distances = []
        labels = []
        for lookup in lookups:
            distances.append(lookup[0]['distance'])
            labels.append(lookup[0]['label'])

        if verbose:
            pb = tqdm(total=len(distances) * len(self.cutpoints),
                      desc='matching embeddings')

        matches = defaultdict(list)
        for name, cp in self.cutpoints.items():
            threshold = float(cp['distance'])
            for idx, distance in enumerate(distances):
                if distance <= threshold:
                    label = labels[idx]
                else:
                    label = no_match_label
                matches[name].append(label)

                if verbose:
                    pb.update()

        if verbose:
            pb.close()

        return matches

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
            "embedding_output": self.embedding_output,
            "stat_buffer_size": self.stat_buffer_size,
            "is_calibrated": self.is_calibrated,
            "calibration_metric_config": self.calibration_metric.get_config(),
            "cutpoints": self.cutpoints,
            "calibration_thresholds": self.calibration_thresholds
        }

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
                        embedding_output=md['embedding_output'],
                        evaluator=md['evaluator'],
                        match_algorithm=md['match_algorithm'],
                        stat_buffer_size=md['stat_buffer_size'])

        # reload the tables
        if verbose:
            print("Loading index data")
        index.table.load(path)

        # rebuild the index
        if verbose:
            print('Loading index matcher')
        index.matcher.load(path)

        # reload calibration data if any

        index.is_calibrated = md['is_calibrated']
        if index.is_calibrated:
            if verbose:
                print("Loading calibration data")
            index.calibration_metric = EvalMetric.from_config(md['calibration_metric_config'])  # noqa
            index.cutpoints = md['cutpoints']
            index.calibration_thresholds = md['calibration_thresholds']
        return index

    def get_calibration_metric(self):
        return self.calibration_metric

    def size(self) -> int:
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
            ['index size', self.size()],
            ['calibrated', self.is_calibrated],
            ['calibration_metric', self.calibration_metric],
            ['embedding_output', self.embedding_output]
        ]
        print(tabulate(rows))
        print('\n')

        print('\n[Performance]')
        rows = [['num lookups', stats['num_lookups']]]
        for k, v in stats['query_performance'].items():
            rows.append([k, v])
        print(tabulate(rows))

    def to_data_frame(self, num_items: int = 0) -> PandasDataFrame:
        """Export data as pandas dataframe

        Args:
            num_items (int, optional): Num items to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """
        return self.table.to_data_frame(num_items)

    @staticmethod
    def __make_metadata_fname(path):
        return str(Path(path) / 'index_metadata.json')
