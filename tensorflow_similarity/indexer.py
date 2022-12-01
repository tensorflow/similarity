# Copyright 2021 The TensorFlow Authors
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

"""Index the embeddings infered by the model to allow distance based
sub-linear search"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tqdm.auto import tqdm

from .classification_metrics import (
    ClassificationMetric,
    F1Score,
    make_classification_metric,
)

# internal
from .distances import Distance, distance_canonicalizer
from .evaluators import Evaluator, MemoryEvaluator
from .matchers import ClassificationMatch, make_classification_matcher
from .retrieval_metrics import RetrievalMetric
from .search import NMSLibSearch, Search, make_search
from .stores import MemoryStore, Store
from .types import CalibrationResults, FloatTensor, Lookup, PandasDataFrame, Tensor
from .utils import unpack_lookup_distances, unpack_lookup_labels


class Indexer:
    """Indexing system that allows to efficiently find nearest embeddings
    by indexing known embeddings and make them searchable using an
    [Approximate Nearest Neighbors Search]
    (https://en.wikipedia.org/wiki/Nearest_neighbor_search)
    search implemented via the [`Search()`](search/overview.md) classes
    and associated data lookup via the [`Store()`](stores/overview.md) classes.

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

    def __init__(
        self,
        embedding_size: int,
        distance: Distance | str = "cosine",
        search: Search | str = "nmslib",
        kv_store: Store | str = "memory",
        evaluator: Evaluator | str = "memory",
        embedding_output: int | None = None,
        stat_buffer_size: int = 1000,
    ) -> None:
        """Index embeddings to make them searchable via KNN

        Args:
            embedding_size: Size of the embeddings that will be stored.
            It is usually equivalent to the size of the output layer.

            distance: Distance used to compute embeddings proximity.
            Defaults to 'cosine'.

            kv_store: How to store the indexed records.
            Defaults to 'memory'.

            search: Which `Search()` framework to use to perform KNN
            search. Defaults to 'nmslib'.

            evaluator: What type of `Evaluator()` to use to evaluate index
            performance. Defaults to in-memory one.

            embedding_output: Which model output head predicts the embeddings
            that should be indexed. Default to None which is for single output
            model. For multi-head model, the callee, usually the
            `SimilarityModel()` class is responsible for passing the correct
            one.

            stat_buffer_size: Size of the sliding windows
            buffer used to compute index performance. Defaults to 1000.

        Raises:
            ValueError: Invalid search framework or key value store.
        """
        distance = distance_canonicalizer(distance)
        self.distance = distance  # needed for save()/load()
        self.embedding_output = embedding_output
        self.embedding_size = embedding_size

        # internal structure naming
        # FIXME support custom objects
        self.search_type = search
        self.kv_store_type = kv_store
        self.evaluator_type = evaluator

        # stats configuration
        self.stat_buffer_size = stat_buffer_size

        # calibration
        self.is_calibrated = False
        self.calibration_metric: ClassificationMetric = F1Score()
        self.cutpoints: Mapping[str, Mapping[str, float | str]] = {}
        self.calibration_thresholds: Mapping[str, np.ndarray] = {}

        # initialize internal structures
        self._init_structures()

    def reset(self) -> None:
        "Reinitialize the indexer"
        self._init_structures()

    def _init_structures(self) -> None:
        "(re)initialize internal storage structure"

        if self.search_type == "nmslib":
            self.search: Search = NMSLibSearch(distance=self.distance, dim=self.embedding_size)
        elif isinstance(self.search_type, Search):
            self.search = self.search_type
        else:
            raise ValueError("You need to either supply a known search " "framework name or a Search() object")

        # mapper from id to record data
        if self.kv_store_type == "memory":
            self.kv_store: Store = MemoryStore()
        elif isinstance(self.kv_store_type, Store):
            self.kv_store = self.kv_store_type
        else:
            raise ValueError("You need to either supply a know key value " "store name or a Store() object")

        # code used to evaluate indexer performance
        if self.evaluator_type == "memory":
            self.evaluator: Evaluator = MemoryEvaluator()
        elif isinstance(self.evaluator_type, Evaluator):
            self.evaluator = self.evaluator_type
        else:
            raise ValueError("You need to either supply a know evaluator name " "or an Evaluator() object")

        # stats
        self._stats: defaultdict[str, int] = defaultdict(int)
        self._lookup_timings_buffer: deque[float] = deque([], maxlen=self.stat_buffer_size)

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
            prediction: TF similarity model prediction, may be a multi-headed
            output.

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
            predictions: TF similarity model predictions, may be a multi-headed
            output.

        Returns:
            Tensor: 2D Tensor (num_embeddings, embedding_value)
        """

        if isinstance(self.embedding_output, int):
            embeddings: FloatTensor = predictions[self.embedding_output]
        else:
            # needed for typing
            embeddings = predictions
        return embeddings

    def _cast_label(self, label: int | None) -> int | None:
        if label is not None:
            label = int(label)
        return label

    def add(
        self,
        prediction: FloatTensor,
        label: int | None = None,
        data: Tensor = None,
        build: bool = True,
        verbose: int = 1,
    ):
        """Add a single embedding to the indexer

        Args:
            prediction: TF similarity model prediction, may be a multi-headed
            output.

            label: Label(s) associated with the
            embedding. Defaults to None.

            data: Input data associated with
            the embedding. Defaults to None.

            build: Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batches/points and build it manually once after.

            verbose: Display progress if set to 1.
            Defaults to 1.
        """

        # deal with potential multi-output
        embedding = self._get_embedding(prediction)

        # store data and get its id
        idx = self.kv_store.add(embedding, label, data)

        # add index to the embedding
        self.search.add(embedding, idx, build=build, verbose=verbose)

    def batch_add(
        self,
        predictions: FloatTensor,
        labels: Sequence[int] | None = None,
        data: Tensor | None = None,
        build: bool = True,
        verbose: int = 1,
    ):
        """Add a batch of embeddings to the indexer

        Args:
            predictions: TF similarity model predictions, may be a multi-headed
            output.

            labels: label(s) associated with the embedding. Defaults to None.

            datas: input data associated with the embedding. Defaults to None.

            build: Rebuild the index after insertion.
            Defaults to True. Set it to false if you would like to add
            multiples batches/points and build it manually once after.

            verbose: Display progress if set to 1. Defaults to 1.
        """

        # deal with potential multi-output
        embeddings = self._get_embeddings(predictions)

        # store points
        if verbose:
            print("|-Storing data points in key value store")
        idxs = self.kv_store.batch_add(embeddings, labels, data)
        self.search.batch_add(embeddings, idxs, build=build, verbose=verbose)

    def single_lookup(self, prediction: FloatTensor, k: int = 5) -> list[Lookup]:
        """Find the k closest matches of a given embedding

        Args:
            prediction: TF similarity model prediction, may be a multi-headed
            output.

            k: Number of nearest neighbors to lookup. Defaults to 5.
        Returns
            list of the k nearest neighbors info:
            list[Lookup]
        """

        embedding = self._get_embedding(prediction)
        start = time()
        idxs, distances = self.search.lookup(embedding, k=k)
        nn_embeddings, labels, data = self.kv_store.batch_get(idxs)

        lookup_time = time() - start
        lookups = []
        for i in range(len(nn_embeddings)):
            # ! casting is needed to avoid slowness down the line
            lookups.append(
                Lookup(
                    rank=i + 1,
                    embedding=nn_embeddings[i],
                    distance=float(distances[i]),
                    label=self._cast_label(labels[i]),
                    data=data[i],
                )
            )
        self._lookup_timings_buffer.append(lookup_time)
        self._stats["num_lookups"] += 1
        return lookups

    def batch_lookup(self, predictions: FloatTensor, k: int = 5, verbose: int = 1) -> list[list[Lookup]]:

        """Find the k closest matches for a set of embeddings

        Args:
            predictions: TF similarity model predictions, may be a multi-headed
            output.

            k: Number of nearest neighbors to lookup. Defaults to 5.

            verbose: Be verbose. Defaults to 1.

        Returns
            list of list of k nearest neighbors:
            list[list[Lookup]]
        """

        embeddings = self._get_embeddings(predictions)
        num_embeddings = len(embeddings)
        start = time()
        batch_lookups = []

        if verbose:
            print("\nPerforming NN search\n")
        batch_idxs, batch_distances = self.search.batch_lookup(embeddings, k=k)

        if verbose:
            pb = tqdm(total=num_embeddings, desc="Building NN list")
        for eidx in range(num_embeddings):
            lidxs = batch_idxs[eidx]  # list of nn idxs
            distances = batch_distances[eidx]

            nn_embeddings, labels, data = self.kv_store.batch_get(lidxs)
            lookups = []
            for i in range(len(nn_embeddings)):
                # ! casting is needed to avoid slowness down the line
                lookups.append(
                    Lookup(
                        rank=i + 1,
                        embedding=nn_embeddings[i],
                        distance=float(distances[i]),
                        label=self._cast_label(labels[i]),
                        data=data[i],
                    )
                )
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
        self._stats["num_lookups"] += num_embeddings

        return batch_lookups

    # evaluation related functions
    def evaluate_retrieval(
        self,
        predictions: FloatTensor,
        target_labels: Sequence[int],
        retrieval_metrics: Sequence[RetrievalMetric],
        verbose: int = 1,
    ) -> dict[str, np.ndarray]:
        """Evaluate the quality of the index against a test dataset.

        Args:
            predictions: TF similarity model predictions, may be a multi-headed
            output.

            target_labels: Sequence of the expected labels associated with the
            embedded queries.

            retrieval_metrics: list of
            [RetrievalMetric()](retrieval_metrics/overview.md) to compute.

            verbose (int, optional): Display results if set to 1 otherwise
            results are returned silently. Defaults to 1.

        Returns:
            Dictionary of metric results where keys are the metric names and
            values are the metrics values.
        """
        # Determine the maximum number of neighbors needed by the retrieval
        # metrics because we do a single lookup.
        k = 1
        for m in retrieval_metrics:
            if not isinstance(m, RetrievalMetric):
                raise ValueError(
                    m,
                    "is not a valid RetrivalMetric(). The "
                    "RetrivialMetric() must be instantiated with "
                    "a valid K.",
                )
            if m.k > k:
                k = m.k

        # Add one more K to handle the case where we drop the closest lookup.
        # This ensures that we always have enough lookups in the result set.
        k += 1

        # Find NN
        lookups = self.batch_lookup(predictions, k=k, verbose=verbose)

        # Evaluate them
        return self.evaluator.evaluate_retrieval(
            retrieval_metrics=retrieval_metrics,
            target_labels=target_labels,
            lookups=lookups,
        )

    def evaluate_classification(
        self,
        predictions: FloatTensor,
        target_labels: Sequence[int],
        distance_thresholds: Sequence[float] | FloatTensor,
        metrics: Sequence[str | ClassificationMetric] = ["f1"],
        matcher: str | ClassificationMatch = "match_nearest",
        k: int = 1,
        verbose: int = 1,
    ) -> dict[str, np.ndarray]:
        """Evaluate the classification performance.

        Compute the classification metrics given a set of queries, lookups, and
        distance thresholds.

        Args:
            predictions: TF similarity model predictions, may be a multi-headed
            output.

            target_labels: Sequence of expected labels for the lookups.

            distance_thresholds: A 1D tensor denoting the distances points at
            which we compute the metrics.

            metrics: The set of classification metrics.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            distance_rounding: How many digit to consider to
            decide if the distance changed. Defaults to 8.

            verbose: Be verbose. Defaults to 1.
        Returns:
            A Mapping from metric name to the list of values computed for each
            distance threshold.
        """
        combined_metrics: list[ClassificationMetric] = [make_classification_metric(m) for m in metrics]

        lookups = self.batch_lookup(predictions, k=k, verbose=verbose)

        # we also convert to np.ndarray first to avoid a slow down if
        # convert_to_tensor is called on a list.
        query_labels = tf.convert_to_tensor(np.array(target_labels))

        # TODO(ovallis): The float type should be derived from the model.
        lookup_distances = unpack_lookup_distances(lookups, dtype="float32")
        lookup_labels = unpack_lookup_labels(lookups, dtype=query_labels.dtype)
        thresholds: FloatTensor = tf.cast(
            tf.convert_to_tensor(distance_thresholds),
            dtype=lookup_distances.dtype,
        )

        results = self.evaluator.evaluate_classification(
            query_labels=query_labels,
            lookup_labels=lookup_labels,
            lookup_distances=lookup_distances,
            distance_thresholds=thresholds,
            metrics=combined_metrics,
            matcher=matcher,
            verbose=verbose,
        )

        return results

    def calibrate(
        self,
        predictions: FloatTensor,
        target_labels: Sequence[int],
        thresholds_targets: MutableMapping[str, float],
        calibration_metric: str | ClassificationMetric = "f1_score",  # noqa
        k: int = 1,
        matcher: str | ClassificationMatch = "match_nearest",
        extra_metrics: Sequence[str | ClassificationMetric] = [
            "precision",
            "recall",
        ],  # noqa
        rounding: int = 2,
        verbose: int = 1,
    ) -> CalibrationResults:
        """Calibrate model thresholds using a test dataset.

        FIXME: more detailed explanation.

        Args:
            predictions: TF similarity model predictions, may be a multi-headed
            output.

            target_labels: Sequence of the expected labels associated with the
            embedded queries.

            thresholds_targets: Dict of performance targets to (if possible)
            meet with respect to the `calibration_metric`.

            calibration_metric: [ClassificationMetric()](metrics/overview.md)
            used to evaluate the performance of the index.

            k: How many neighbors to use during the calibration.
            Defaults to 1.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.
            Defaults to 'match_nearest'.

            extra_metrics: list of additional
            `tf.similarity.classification_metrics.ClassificationMetric()` to
            compute and report. Defaults to ['precision', 'recall'].

            rounding: Metric rounding. Default to 2 digits.

            verbose: Be verbose and display calibration results. Defaults to 1.

        Returns:
            CalibrationResults containing the thresholds and cutpoints Dicts.
        """

        # find NN
        lookups = self.batch_lookup(predictions, k=k, verbose=verbose)

        # making sure our metrics are all ClassificationMetric objects
        calibration_metric = make_classification_metric(calibration_metric)

        combined_metrics: list[ClassificationMetric] = [make_classification_metric(m) for m in extra_metrics]

        # running calibration
        calibration_results = self.evaluator.calibrate(
            target_labels=target_labels,
            lookups=lookups,
            thresholds_targets=thresholds_targets,
            calibration_metric=calibration_metric,
            matcher=matcher,
            extra_metrics=combined_metrics,
            metric_rounding=rounding,
            verbose=verbose,
        )

        # display cutpoint results if requested
        if verbose:
            headers = ["name", "value", "distance"]  # noqa
            cutpoints = list(calibration_results.cutpoints.values())
            # dynamically find which metrics we need. We only need to look at
            # the first cutpoints dictionary as all subsequent ones will have
            # the same metric keys.
            for metric_name in cutpoints[0].keys():
                if metric_name not in headers:
                    headers.append(metric_name)

            rows = []
            for data in cutpoints:
                rows.append([data[v] for v in headers])
            print("\n", tabulate(rows, headers=headers))

        # store info for serialization purpose
        self.is_calibrated = True
        self.calibration_metric = calibration_metric
        self.cutpoints = calibration_results.cutpoints
        self.calibration_thresholds = calibration_results.thresholds
        return calibration_results

    def match(
        self,
        predictions: FloatTensor,
        no_match_label: int = -1,
        k=1,
        matcher: str | ClassificationMatch = "match_nearest",
        verbose: int = 1,
    ) -> dict[str, list[int]]:
        """Match embeddings against the various cutpoints thresholds

        Args:
            predictions: TF similarity model predictions, may be a multi-headed
            output.

            no_match_label: What label value to assign when there is no match.
            Defaults to -1.

            k: How many neighboors to use during the calibration.
            Defaults to 1.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            verbose: display progression. Default to 1.

        Notes:

            1. It is up to the [`SimilarityModel.match()`](similarity_model.md)
            code to decide which of cutpoints results to use / show to the
            users. This function returns all of them as there is little
            performance downside to do so and it makes the code clearer
            and simpler.

            2. The calling function is responsible to return the list of class
            matched to allows implementation to use additional criteria if they
            choose to.

        Returns:
            Dict of cutpoint names mapped to lists of matches.
        """
        matcher = make_classification_matcher(matcher)

        lookups = self.batch_lookup(predictions, k=k, verbose=verbose)

        lookup_distances = unpack_lookup_distances(lookups, dtype=predictions.dtype)
        # TODO(ovallis): The int type should be derived from the model.
        lookup_labels = unpack_lookup_labels(lookups, dtype="int32")

        if verbose:
            pb = tqdm(
                total=len(lookup_distances) * len(self.cutpoints),
                desc="matching embeddings",
            )

        matches: defaultdict[str, list[int]] = defaultdict(list)
        for cp_name, cp_data in self.cutpoints.items():
            distance_threshold = float(cp_data["distance"])

            pred_labels, pred_dist = matcher.derive_match(
                lookup_labels=lookup_labels, lookup_distances=lookup_distances
            )

            for label, distance in zip(pred_labels, pred_dist):
                if distance <= distance_threshold:
                    label = int(label)
                else:
                    label = no_match_label

                matches[cp_name].append(label)

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
            "kv_store": self.kv_store_type,
            "evaluator": self.evaluator_type,
            "search_config": self.search.get_config(),
            "stat_buffer_size": self.stat_buffer_size,
            "is_calibrated": self.is_calibrated,
            "calibration_metric_config": self.calibration_metric.get_config(),
            "cutpoints": self.cutpoints,
            # convert np.arrays to list before serialization
            "calibration_thresholds": {k: v.tolist() for k, v in self.calibration_thresholds.items()},
        }

        metadata_fname = self.__make_metadata_fname(path)
        tf.io.write_file(metadata_fname, json.dumps(metadata))

        self.kv_store.save(path, compression=compression)
        self.search.save(path)

    @staticmethod
    def load(path: str | Path, verbose: int = 1):
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
        search = make_search(md["search_config"])
        index = Indexer(
            distance=md["distance"],
            embedding_size=md["embedding_size"],
            embedding_output=md["embedding_output"],
            kv_store=md["kv_store"],
            evaluator=md["evaluator"],
            search=search,
            stat_buffer_size=md["stat_buffer_size"],
        )

        # reload the key value store
        if verbose:
            print("Loading index data")
        index.kv_store.load(path)

        # rebuild the index
        if verbose:
            print("Loading search index")
        index.search.load(path)

        # reload calibration data if any
        index.is_calibrated = md["is_calibrated"]
        if index.is_calibrated:
            if verbose:
                print("Loading calibration data")
            index.calibration_metric = make_classification_metric(
                metric=md["calibration_metric_config"]["canonical_name"],
                name=md["calibration_metric_config"]["name"],
            )

            index.cutpoints = md["cutpoints"]
            index.calibration_thresholds = {k: np.array(v) for k, v in md["calibration_thresholds"].items()}

        return index

    def get_calibration_metric(self):
        return self.calibration_metric

    def size(self) -> int:
        "Return the index size"
        return self.kv_store.size()

    def stats(self):
        """return index statistics"""
        stats = self._stats
        stats["size"] = self.kv_store.size()
        stats["stat_buffer_size"] = self.stat_buffer_size

        # query performance - make sure we don't count unused buffer
        max_idx = min(stats["num_lookups"], self.stat_buffer_size)
        lookup_timings = list(self._lookup_timings_buffer)[:max_idx]

        # ensure we never have an empty list
        lookup_timings = lookup_timings if list(lookup_timings) else [0]

        # compute stats
        stats["query_performance"] = {
            "min": np.min(lookup_timings),
            "max": np.max(lookup_timings),
            "avg": np.average(lookup_timings),
            "median": np.median(lookup_timings),
            "stddev": np.std(lookup_timings),
        }
        return stats

    def print_stats(self):
        "display statistics in terminal friendly fashion"
        # compute statistics
        stats = self.stats()

        # info
        print("[Info]")
        rows = [
            ["distance", self.distance],
            ["key value store", self.kv_store_type],
            ["search algorithm", self.search_type],
            ["evaluator", self.evaluator_type],
            ["index size", self.size()],
            ["calibrated", self.is_calibrated],
            ["calibration_metric", self.calibration_metric.name],
            ["embedding_output", self.embedding_output],
        ]
        print(tabulate(rows))
        print("\n")

        print("\n[Performance]")
        rows = [["num lookups", stats["num_lookups"]]]
        for k, v in stats["query_performance"].items():
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
        return self.kv_store.to_data_frame(num_items)

    @staticmethod
    def __make_metadata_fname(path):
        return str(Path(path) / "index_metadata.json")
