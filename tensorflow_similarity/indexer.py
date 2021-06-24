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
from .types import FloatTensor, Lookup, PandasDataFrame, Tensor

# internal
from .distances import distance_canonicalizer, Distance
from .matchers import Matcher, NMSLibMatcher
from .tables import Table, MemoryTable
from .evaluators import Evaluator, MemoryEvaluator
from .metrics import EvalMetric, make_metric, F1Score


class Indexer():
    """Indexing system that allows to efficiently find nearest embeddings
    by indexing known embeddings and make them searchable using an
    [Approximate Nearest Neigboors Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
    search implemented via the [`Matcher()`](matchers/overview.md) classes
    and associated data lookup via the [`Table()`](tables/overview.md) classes.

    The indexer allows to evaluate the quality of the constructed index and
    calibrate the [SimilarityModel.match()](similarity_model.md) function via
    the [`Evaluator()`](evaluators/overview.md) classes.
    """
    # semantic sugar for the order of the returned data
    EMBEDDINGS = 0
    DISTANCES = 1
    LABELS = 2
    DATA = 3
    RANKS = 4

    def __init__(self,
                 embedding_size: int,
                 distance: Union[Distance, str] = 'cosine',
                 matcher: Union[Matcher, str] = 'nmslib',
                 table: Union[Table, str] = 'memory',
                 evaluator: Union[Evaluator, str] = 'memory',
                 embedding_output: int = None,
                 stat_buffer_size: int = 1000) -> None:
        """Index embeddings to make them searchable via KNN

        Args:
            embedding_size: Size of the embeddings that will be stored.
            It is usually equivalent to the size of the output layer.

            distance: Distance used to compute embeddings proximity.
            Defaults to 'cosine'.

            table: How to store the index records.
            Defaults to 'memory'.

            matcher: Which `Matcher()` framework to use to perfom KNN
            search. Defaults to 'nmslib'.

            evaluator: What type of `Evaluator()` to use to evaluate index
            performance. Defaults to in-memory one.

            embedding_output: Which model output head predicts
            the embbedings that should be indexed. Default to None which is for
            single output model. For multi-head model, the callee, usually the
            `SimilarityModel()` class is responsible for passing the
            correct one.

            stat_buffer_size: Size of the sliding windows
            buffer used to computer index performance. Defaults to 1000.

        Raises:
            ValueError: Invalid matcher or table.
        """
        distance = distance_canonicalizer(distance)
        self.distance = distance  # needed for save()/load()
        self.embedding_output = embedding_output
        self.embedding_size = embedding_size

        # internal structure naming
        # FIXME support custom objects
        self.matcher_type = matcher
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

        if self.matcher_type == 'nmslib':
            self.matcher: Matcher = NMSLibMatcher(distance=self.distance,
                                                  dims=self.embedding_size)
        elif isinstance(self.matcher_type, Matcher):
            self.matcher = self.matcher_type
        else:
            raise ValueError("You need to either supply a known matcher name\
                or a Matcher() object")

        # mapper from id to record data
        if self.table_type == 'memory':
            self.table: Table = MemoryTable()
        elif isinstance(self.table_type, Table):
            self.table = self.table_type
        else:
            raise ValueError("You need to either supply a know table name\
                or a Table() object")

        # code used to evaluate indexer performance
        if self.evaluator_type == 'memory':
            self.evaluator: Evaluator = MemoryEvaluator()
        elif isinstance(self.evaluator_type, Evaluator):
            self.evaluator = self.evaluator_type
        else:
            raise ValueError("You need to either supply a know evaluator name\
                or an Evaluator() object")
        # stats
        self._stats: DefaultDict[str, int] = defaultdict(int)
        self._lookup_timings_buffer: Deque = deque([], maxlen=self.stat_buffer_size)  # noqa

        # calibration data
        self.is_calibrated = False
        self.calibration_metric = F1Score()
        self.cutpoints = {}
        self.calibration_thresholds = {}

    def _get_embedding(self, prediction: FloatTensor) -> FloatTensor:
        """Return the 1st embedding vector from a (multi-output) model
        prediction

        See: `single_lookup()`, `add()`

        Args:
            prediction: Model prediction for single embedding.

        Returns:
            FloatTensor: 1D Tensor that contains the actual embedding
        """

        if isinstance(self.embedding_output, int):
            # in multi-output: embedding is [output_num][0]
            embedding: FloatTensor = prediction[self.embedding_output][0]
        else:
            # single output > return 1st element
            embedding = prediction[0]
        return embedding

    def _get_embeddings(self, predictions: FloatTensor) -> FloatTensor:
        """Return the embedding vectors from a (multi-output) model prediction

        Args:
            prediction: Model predictions.

        Returns:
            Tensor: 2D Tensor (num_embeddings, embedding_value)
        """

        if isinstance(self.embedding_output, int):
            embeddings: FloatTensor = predictions[self.embedding_output]
        else:
            # needed for typing
            embeddings = predictions
        return embeddings

    def _cast_label(self, label: Optional[int]) -> Optional[int]:
        if label is not None:
            label = int(label)
        return label

    def add(self,
            prediction: FloatTensor,
            label: Optional[int] = None,
            data: Tensor = None,
            build: bool = True,
            verbose: int = 1):
        """ Add a single embedding to the indexer

        Args:
            prediction: TF similarity model prediction.

            label: Label(s) associated with the
            embedding. Defaults to None.

            data: Input data associated with
            the embedding. Defaults to None.

            build: Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batchs/points and build it manually once after.

            verbose: Display progress if set to 1.
            Defaults to 1.
        """

        # deal with potential multi-output
        embedding = self._get_embedding(prediction)

        # store data and get its id
        idx = self.table.add(embedding, label, data)

        # add index to the embedding
        self.matcher.add(embedding, idx, build=build, verbose=verbose)

    def batch_add(self,
                  predictions: FloatTensor,
                  labels: Optional[List[Optional[int]]] = None,
                  data: Optional[Tensor] = None,
                  build: bool = True,
                  verbose: int = 1):
        """Add a batch of embeddings to the indexer

        Args:
            predictions: TF similarity model predictions.

            labels: label(s) associated with the embedding. Defaults to None.

            datas: input data associated with the embedding. Defaults to None.

            build: Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batchs/points and build it manually once after.

            verbose: Display progress if set to 1. Defaults to 1.
        """

        # deal with potential multi-output
        embeddings = self._get_embeddings(predictions)

        # store points
        if verbose:
            print('|-Storing data points in index table')
        idxs = self.table.batch_add(embeddings, labels, data)
        self.matcher.batch_add(embeddings, idxs, build=build, verbose=verbose)

    def single_lookup(
            self,
            prediction: FloatTensor,
            k: int = 5) -> List[Lookup]:
        """Find the k closest matches of a given embedding

        Args:
            prediction: model prediction.
            k: Number of nearest neighboors to lookup. Defaults to 5.
        Returns
            list of the k nearest neigboors info:
            List[Lookup]
        """

        embedding = self._get_embedding(prediction)
        start = time()
        idxs, distances = self.matcher.lookup(embedding, k=k)
        nn_embeddings, labels, data = self.table.batch_get(idxs)

        lookup_time = time() - start
        lookups = []
        for i in range(len(nn_embeddings)):
            # ! casting is needed to avoid slowness down the line
            lookups.append(Lookup(
                rank=i + 1,
                embedding=nn_embeddings[i],
                distance=float(distances[i]),
                label=self._cast_label(labels[i]),
                data=data[i]
                ))
        self._lookup_timings_buffer.append(lookup_time)
        self._stats['num_lookups'] += 1
        return lookups

    def batch_lookup(self,
                     predictions: FloatTensor,
                     k: int = 5,
                     verbose: int = 1) -> List[List[Lookup]]:

        """Find the k closest matches for a set of embeddings

        Args:
            predictions: model predictions.
            k: Number of nearest neighboors to lookup. Defaults to 5.
            verbose: Be verbose. Defaults to 1.

        Returns
            list of list of k nearest neighboors:
            List[List[Lookup]]
        """

        embeddings = self._get_embeddings(predictions)
        num_embeddings = len(embeddings)
        start = time()
        batch_lookups = []

        if verbose:
            print("\nPerforming NN search\n")
        batch_idxs, batch_distances = self.matcher.batch_lookup(embeddings,
                                                                k=k)

        if verbose:
            pb = tqdm(total=num_embeddings, desc='Building NN list')
        for eidx in range(num_embeddings):
            lidxs = batch_idxs[eidx]   # list of nn idxs
            distances = batch_distances[eidx]

            nn_embeddings, labels, data = self.table.batch_get(lidxs)
            lookups = []
            for i in range(len(nn_embeddings)):
                # ! casting is needed to avoid slowness down the line
                lookups.append(Lookup(
                    rank=i + 1,
                    embedding=nn_embeddings[i],
                    distance=float(distances[i]),
                    label=self._cast_label(labels[i]),
                    data=data[i]
                ))
            batch_lookups.append(lookups)

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

        return batch_lookups

    # evaluation related functions
    def evaluate(self,
                 predictions: FloatTensor,
                 y: List[int],
                 metrics: List[Union[str, EvalMetric]],
                 k: int = 1,
                 verbose: int = 1) -> Dict[str, Union[float, int]]:
        """Evaluate the quality of the index against a test dataset.

        Args:
            predictions: Test emebddings computed by the SimilarityModel.
            y: Expected labels for the nearest neighboors.
            metrics: List of [Metric()](metrics/overview.md) to compute.
            k: How many neighboors to use during the evaluation. Defaults to 1.
            verbose: Be verbose. Defaults to 1.

        Returns:
            Dictionary of metric results where keys are the metric names and
            values are the metrics values.
        """
        # Find NN
        lookups = self.batch_lookup(predictions, verbose=verbose)

        # Evaluate them
        return self.evaluator.evaluate(index_size=self.size(),
                                       metrics=metrics,
                                       targets_labels=y,
                                       lookups=lookups)

    def calibrate(self,
                  predictions: FloatTensor,
                  y: List[int],
                  thresholds_targets: Dict[str, float],
                  calibration_metric: Union[str, EvalMetric] = "f1_score",
                  k: int = 1,
                  extra_metrics: List[Union[str, EvalMetric]] = ['accuracy', 'recall'],  # noqa
                  rounding: int = 2,
                  verbose: int = 1
                  ) -> Dict[str, Union[Dict[str, float], List[float]]]:
        """Calibrate model thresholds using a test dataset.

        FIXME: more detailed explaination.

        Args:

            predictions: Test emebddings computed by the SimilarityModel.

            y: Expected labels for the nearest neighboors.

            thresholds_targets: Dict of performance targets to (if possible)
            meet with respect to the `calibration_metric`.

            calibration_metric: [Metric()](metrics/overview.md) used to
            evaluate the performance of the index.

            k: How many neighboors to use during the calibration.
            Defaults to 1.

            extra_metrics: List of additional [Metric()](metrics/overview.md)
            to compute and report.

            rounding: Metric rounding. Default to 2 digits.

            verbose: Be verbose and display calibration results. Defaults to 1.

        Returns:
           Calibration results: `{"cutpoints": {}, "thresholds": {}}`
        """

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
            print("\n", tabulate(rows, headers=headers))

        # store info for serialization purpose
        self.is_calibrated = True
        self.calibration_metric = calibration_metric
        self.cutpoints = cutpoints
        self.calibration_thresholds = thresholds
        return {"cutpoints": cutpoints, "thresholds": thresholds}

    def match(self,
              predictions: FloatTensor,
              no_match_label: int = -1,
              verbose: int = 1) -> Dict[str, List[int]]:
        """Match embeddings against the various cutpoints thresholds

        Args:
            predictions (FloatTensor): embeddings

            no_match_label (int, optional): What label value to assign when
            there is no match. Defaults to -1.

            verbose (int): display progression. Default to 1.

        Notes:

            1. It is up to the [`SimilarityModel.match()`](similarity_model.md)
            code to decide which of cutpoints results to use / show to the
            users. This function returns all of them as there is little
            performance downside to do so and it makes the code clearer
            and simpler.

            2. The calling function is responsible to return the list of class
            matched to allows implementation to use additional criterias
            if they choose to.

        Returns:
            Dict of matches list keyed by cutpoint names.
        """
        lookups = self.batch_lookup(predictions, k=1, verbose=verbose)

        # vectorize
        distances = []
        labels = []
        for lookup in lookups:
            distances.append(lookup[0].distance)
            labels.append(lookup[0].label)

        if verbose:
            pb = tqdm(total=len(distances) * len(self.cutpoints),
                      desc='matching embeddings')

        matches: Dict = defaultdict(list)
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

    def save(self, path: str, compression: bool = True):
        """Save the index to disk

        Args:
            path: directory where to save the index
            compression: Store index data compressed. Defaults to True.
        """
        path = str(path)
        # saving metadata
        metadata = {
            "size": self.size(),
            "compression": compression,
            "distance": self.distance.name,
            "embedding_output": self.embedding_output,
            "embedding_size": self.embedding_size,

            "table": self.table_type,
            "evaluator": self.evaluator_type,
            "matcher": self.matcher_type,

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
    def load(path: Union[str, Path], verbose: int = 1):
        """Load Index data from a checkpoint and initialize underlying
        structure with the reloaded data.

        Args:
            path: Directory where the checkpoint is located.
            verbose: Be verbose. Defaults to 1.

        Returns:
            Initialized index
        """
        path = str(path)
        # recreate the index from metadata
        metadata_fname = Indexer.__make_metadata_fname(path)
        metadata = tf.io.read_file(metadata_fname)
        metadata = tf.keras.backend.eval(metadata)
        md = json.loads(metadata)
        index = Indexer(distance=md['distance'],
                        embedding_size=md['embedding_size'],
                        embedding_output=md['embedding_output'],
                        table=md['table'],
                        evaluator=md['evaluator'],
                        matcher=md['matcher'],
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
            ['matching algorithm', self.matcher_type],
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
