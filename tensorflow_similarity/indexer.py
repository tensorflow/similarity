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
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tqdm.auto import tqdm

# internal
import tensorflow_similarity.search
import tensorflow_similarity.stores

from .base_indexer import BaseIndexer
from .classification_metrics import F1Score, make_classification_metric
from .types import Lookup

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tensorflow_similarity.distances import Distance
    from tensorflow_similarity.search import Search
    from tensorflow_similarity.stores import Store

    from .evaluators import Evaluator
    from .types import FloatTensor, PandasDataFrame, Tensor


class Indexer(BaseIndexer):
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
        search: Search | str = "linear",
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
            search. Defaults to 'linear'.

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
        super().__init__(distance, embedding_output, embedding_size, evaluator, stat_buffer_size)

        self.search: Search = tensorflow_similarity.search.get(search, distance=distance, dim=embedding_size)
        self.kv_store: Store = tensorflow_similarity.stores.get(kv_store)

        # initialize internal structures
        self._init_structures()

    def reset(self) -> None:
        "Reinitialize the indexer"
        self.search.reset()
        self.kv_store.reset()
        self._init_structures()

    def _init_structures(self) -> None:
        "(re)initialize internal stats structure"

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

    def save(self, path: Path | str, compression: bool = True, overwrite: bool = True):
        """Save the index to disk

        Args:
            path: directory where to save the index
            compression: Store index data compressed. Defaults to True.
            overwrite: Overwrite previous index. Defaults to True.
        """
        path = Path(path)

        if path.exists() and overwrite:
            tf.io.gfile.rmtree(path)

        # saving metadata
        metadata = {
            "size": self.size(),
            "compression": compression,
            "distance": self.distance.name,
            "embedding_output": self.embedding_output,
            "embedding_size": self.embedding_size,
            "kv_store_config": tensorflow_similarity.stores.serialize(self.kv_store),
            "evaluator": self.evaluator_type,
            "search_config": tensorflow_similarity.search.serialize(self.search),
            "stat_buffer_size": self.stat_buffer_size,
            "is_calibrated": self.is_calibrated,
            "calibration_metric_config": self.calibration_metric.get_config(),
            "cutpoints": self.cutpoints,
            # convert np.arrays to list before serialization
            "calibration_thresholds": {k: v.tolist() for k, v in self.calibration_thresholds.items()},
        }

        metadata_fname = self.__make_metadata_fname(str(path))
        tf.io.write_file(metadata_fname, json.dumps(metadata))

        (path / "store").mkdir(parents=True)
        (path / "search").mkdir(parents=True)
        self.kv_store.save(path / "store", compression=compression)
        self.search.save(path / "search")

    @staticmethod
    def load(path: Path | str, verbose: int = 1):
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
        search = tensorflow_similarity.search.get(md["search_config"])
        kv_store = tensorflow_similarity.stores.get(md["kv_store_config"])
        index = Indexer(
            distance=md["distance"],
            embedding_size=md["embedding_size"],
            embedding_output=md["embedding_output"],
            kv_store=kv_store,
            evaluator=md["evaluator"],
            search=search,
            stat_buffer_size=md["stat_buffer_size"],
        )

        # reload the key value store
        if verbose:
            print("Loading index data")
        index.kv_store.load(str(Path(path) / "store"))

        # rebuild the index
        if verbose:
            print("Loading search index")
        index.search.load(str(Path(path) / "search"))

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
            ["key value store", self.kv_store.__class__.__name__],
            ["search algorithm", self.search.__class__.__name__],
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
