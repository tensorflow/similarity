from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tqdm.auto import tqdm

from .classification_metrics import (
    ClassificationMetric,
    F1Score,
    make_classification_metric,
)
from .distances import Distance, distance_canonicalizer
from .evaluators import Evaluator, MemoryEvaluator
from .matchers import ClassificationMatch, make_classification_matcher
from .retrieval_metrics import RetrievalMetric
from .types import CalibrationResults, FloatTensor, Lookup, Tensor
from .utils import unpack_lookup_distances, unpack_lookup_labels


class BaseIndexer(ABC):
    def __init__(
        self,
        distance: Distance | str,
        embedding_output: int | None,
        embedding_size: int,
        evaluator: Evaluator | str,
        stat_buffer_size: int,
    ) -> None:
        distance = distance_canonicalizer(distance)
        self.distance = distance  # needed for save()/load()
        self.embedding_output = embedding_output
        self.embedding_size = embedding_size

        # internal structure naming
        self.evaluator_type = evaluator

        # code used to evaluate indexer performance
        if self.evaluator_type == "memory":
            self.evaluator: Evaluator = MemoryEvaluator()
        elif isinstance(self.evaluator_type, Evaluator):
            self.evaluator = self.evaluator_type
        else:
            raise ValueError("You need to either supply a know evaluator name " "or an Evaluator() object")

        # stats configuration
        self.stat_buffer_size = stat_buffer_size

        # calibration
        self.is_calibrated = False
        self.calibration_metric: ClassificationMetric = F1Score()
        self.cutpoints: Mapping[str, Mapping[str, float | str]] = {}
        self.calibration_thresholds: Mapping[str, np.ndarray] = {}

        return

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
        eval_ret: dict[str, np.ndarray] = self.evaluator.evaluate_retrieval(
            retrieval_metrics=retrieval_metrics,
            target_labels=target_labels,
            lookups=lookups,
        )
        return eval_ret

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
        lookup_distances = unpack_lookup_distances(lookups, dtype=tf.keras.backend.floatx())
        lookup_labels = unpack_lookup_labels(lookups, dtype=query_labels.dtype)
        thresholds: FloatTensor = tf.cast(
            tf.convert_to_tensor(distance_thresholds),
            dtype=tf.keras.backend.floatx(),
        )

        results: dict[str, np.ndarray] = self.evaluator.evaluate_classification(
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
        calibration_results: CalibrationResults = self.evaluator.calibrate(
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
        k: int = 1,
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
