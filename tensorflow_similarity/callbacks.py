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
"""Specialized callbacks that track similarity metrics during training"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from .classification_metrics import make_classification_metric  # noqa
from .classification_metrics import ClassificationMetric
from .evaluators import Evaluator, MemoryEvaluator
from .matchers import ClassificationMatch
from .models import SimilarityModel
from .types import FloatTensor, IntTensor, Tensor
from .utils import unpack_lookup_distances, unpack_lookup_labels, unpack_results


class EvalCallback(Callback):
    """Epoch end evaluation callback that build a test index and evaluate
    model performance on it.

    This evaluation only run at epoch_end as it is computationally very
    expensive.

    """

    def __init__(
        self,
        queries: Tensor,
        query_labels: Sequence[int],
        targets: Tensor,
        target_labels: Sequence[int],
        distance: str = "cosine",
        metrics: Sequence[Union[str, ClassificationMetric]] = [
            "binary_accuracy",
            "f1score",
        ],  # noqa
        tb_logdir: str = None,
        k: int = 1,
        matcher: Union[str, ClassificationMatch] = "match_nearest",
        distance_thresholds: Optional[FloatTensor] = None,
        known_classes: Optional[IntTensor] = None,
    ):
        """Evaluate model matching quality against a validation dataset at
        epoch end.

        Args:
            queries: Test examples that will be tested against the built index.

            query_labels: Queries nearest neighbors expected labels.

            targets: Examples that are indexed.

            target_labels: Target examples labels.

            distance: Distance function used to compute pairwise distance
            between examples embeddings.

            metrics: List of
            'tf.similarity.classification_metrics.ClassificationMetric()` to
            compute during the evaluation. Defaults to ['binary_accuracy',
            'f1score'].

            tb_logdir: Where to write TensorBoard logs. Defaults to None.

            k: The number of nearest neighbors to return for each query.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            distance_thresholds: A 1D tensor denoting the distances points at
            which we compute the metrics. If None, distance_thresholds is set
            to tf.constant([math.inf])

            known_classes: Optional. If specified the validation set will be
            split into two subsets based on whether a class was seen or not
            during training. This should be a list of classes seen during
            training.

        """
        super().__init__()
        t_query_labels = tf.convert_to_tensor(np.array(query_labels))
        self.target_labels = tf.convert_to_tensor(np.array(target_labels))

        # Ensure all label tensors have the same dtype.
        self.target_labels = tf.cast(self.target_labels, dtype=t_query_labels.dtype)

        self.targets = targets
        self.distance = distance
        self.evaluator = MemoryEvaluator()
        # typing requires this weird formulation of creating a new list
        self.metrics: List[ClassificationMetric] = [make_classification_metric(m) for m in metrics]
        self.k = k
        self.matcher = matcher

        if distance_thresholds is not None:
            self.distance_thresholds = distance_thresholds
        else:
            self.distance_thresholds = tf.constant([math.inf])

        if known_classes is not None:
            known_classes = tf.cast(known_classes, dtype=t_query_labels.dtype)
            self.split_validation = True
        else:
            self.split_validation = False

        if tb_logdir:
            tb_logdir = str(Path(tb_logdir) / "index/")
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)
            print("TensorBoard logging enable in %s" % tb_logdir)
        else:
            self.tb_writer = None

        # Use broadcasting to do a y X known_classes equality check. By adding
        # a dim to the start of known_classes and a dim to the end of y, this
        # essentially checks `for ck in known_classes: for cy in y: ck == cy`.
        # We then reduce_any to find all rows in y that match at least one
        # class in known_classes.
        # See https://numpy.org/doc/stable/user/basics.broadcasting.html
        if self.split_validation:
            known_classes = tf.reshape(known_classes, (-1))
            broadcast_classes = tf.expand_dims(known_classes, axis=0)
            broadcast_labels = tf.expand_dims(t_query_labels, axis=-1)
            known_mask = tf.math.reduce_any(broadcast_classes == broadcast_labels, axis=1)
            known_idxs = tf.squeeze(tf.where(known_mask))
            unknown_idxs = tf.squeeze(tf.where(~known_mask))

        with tf.device("/cpu:0"):
            if self.split_validation:
                self.queries_known = tf.gather(queries, indices=known_idxs)
                self.query_labels_known = tf.gather(t_query_labels, indices=known_idxs)
                # Expand to 2D if we only have a single example
                if tf.rank(self.queries_known) == 1:
                    self.queries_known = tf.expand_dims(self.queries_known, axis=0)
                    self.query_labels_known = tf.expand_dims(self.query_labels_known, axis=0)

                self.queries_unknown = tf.gather(queries, indices=unknown_idxs)
                self.query_labels_unknown = tf.gather(t_query_labels, indices=unknown_idxs)
                # Expand to 2D if we only have a single example
                if tf.rank(self.queries_unknown) == 1:
                    self.queries_unknown = tf.expand_dims(self.queries_unknown, axis=0)
                    self.query_labels_unknown = tf.expand_dims(self.query_labels_unknown, axis=0)
            else:
                self.queries_known = queries
                self.query_labels_known = t_query_labels
                self.queries_unknown = None
                self.query_labels_unknown = None

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Computes the eval metrics at the end of each epoch.

        NOTE: This method resets the index and batch adds the target embeddings
        to the index using the new embeddings generated by the current version
        of the model.
        """
        _ = epoch
        if logs is None:
            logs = {}

        # reset the index
        self.model.reset_index()

        # rebuild the index
        self.model.index(self.targets, self.target_labels, verbose=0)

        known_results = _compute_classification_metrics(
            queries=self.queries_known,
            query_labels=self.query_labels_known,
            model=self.model,
            evaluator=self.evaluator,
            metrics=self.metrics,
            k=self.k,
            matcher=self.matcher,
            distance_thresholds=self.distance_thresholds,
        )

        if self.split_validation:
            unknown_results = _compute_classification_metrics(
                queries=self.queries_unknown,
                query_labels=self.query_labels_unknown,
                model=self.model,
                evaluator=self.evaluator,
                metrics=self.metrics,
                k=self.k,
                matcher=self.matcher,
                distance_thresholds=self.distance_thresholds,
            )

            mstr = unpack_results(
                known_results,
                epoch=epoch,
                logs=logs,
                tb_writer=self.tb_writer,
                name_suffix="_known_classes",
            )
            mstr.extend(
                unpack_results(
                    unknown_results,
                    epoch=epoch,
                    logs=logs,
                    tb_writer=self.tb_writer,
                    name_suffix="_unknown_classes",
                )
            )
        else:
            mstr = unpack_results(known_results, epoch=epoch, logs=logs, tb_writer=self.tb_writer)
        self.model.reset_index()
        print(" - ".join(mstr))


def SplitValidationLoss(
    queries: Tensor,
    query_labels: Sequence[int],
    targets: Tensor,
    target_labels: Sequence[int],
    known_classes: IntTensor,
    distance: str = "cosine",
    metrics: Sequence[Union[str, ClassificationMetric]] = [
        "binary_accuracy",
        "f1score",
    ],  # noqa
    tb_logdir: str = None,
    k: int = 1,
    matcher: Union[str, ClassificationMatch] = "match_nearest",
    distance_thresholds: Optional[FloatTensor] = None,
):
    """Creates the validation callbacks.

    Args:
        queries: Test examples that will be tested against the built index.

        query_labels: Queries nearest neighbors expected labels.

        targets: Examples that are indexed.

        target_labels: Target examples labels.

        known_classes: The set of classes seen during training.

        distance: Distance function used to compute pairwise distance
        between examples embeddings.

        metrics: List of
        'tf.similarity.classification_metrics.ClassificationMetric()` to
        compute during the evaluation. Defaults to ['binary_accuracy',
        'f1score'].

        tb_logdir: Where to write TensorBoard logs. Defaults to None.

        k: The number of nearest neighbors to return for each query.
        The lookups are consumed by the Matching Strategy and used to
        derive the matching label and distance.

        matcher: {'match_nearest', 'match_majority_vote'} or
        ClassificationMatch object. Defines the classification matching,
        e.g., match_nearest will count a True Positive if the query_label
        is equal to the label of the nearest neighbor and the distance is
        less than or equal to the distance threshold.

        distance_thresholds: A 1D tensor denoting the distances points at
        which we compute the metrics. If None, distance_thresholds is set
        to tf.constant([math.inf])
    """
    print("WARNING: SplitValidationLoss is deprecated. Please use EvalCallback.")
    return EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        distance=distance,
        metrics=metrics,
        tb_logdir=tb_logdir,
        k=k,
        matcher=matcher,
        distance_thresholds=distance_thresholds,
        known_classes=known_classes,
    )


def _compute_classification_metrics(
    queries: Tensor,
    query_labels: IntTensor,
    model: SimilarityModel,
    evaluator: Evaluator,
    metrics: Sequence[ClassificationMetric],
    k: int,
    matcher: Union[str, ClassificationMatch],
    distance_thresholds: FloatTensor,
) -> Dict[str, np.ndarray]:
    """Compute the classification metrics.

    Args:
        queries: A Tensor of embeddings representing the queries.

        query_labels: An IntTensor representing the class ids associated with
            the queries.

        model: The current similarity model.

        evaluator: An Evalutar object for evaluating the index performance.

        metrics: A list of classification metrics objects.

        k: The number of nearest neighbors to return for each query.

        matcher: {'match_nearest', 'match_majority_vote'} or ClassificationMatch
        object. Defines the classification matching, e.g., match_nearest will
        count a True Positive if the query_label is equal to the label of the
        nearest neighbor and the distance is less than or equal to the distance
        threshold.

        distance_thresholds: A 1D tensor denoting the distances points at which
        we compute the metrics.

    Returns:
        A Python dict mapping the metric name to the copmuted value.
    """
    lookups = model.lookup(queries, k=k, verbose=0)
    lookup_distances = unpack_lookup_distances(lookups, model.dtype)
    lookup_labels = unpack_lookup_labels(lookups, query_labels.dtype)

    # TODO(ovallis): Support passing other matchers. Currently we are using
    # match_nearest.
    results = evaluator.evaluate_classification(
        query_labels=query_labels,
        lookup_labels=lookup_labels,
        lookup_distances=lookup_distances,
        distance_thresholds=distance_thresholds,
        metrics=metrics,
        matcher=matcher,
        verbose=0,
    )

    # The callbacks don't set a distance theshold so we remove it here.
    results.pop("distance")

    return results
