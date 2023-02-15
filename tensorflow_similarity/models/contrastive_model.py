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
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from termcolor import cprint
from tqdm.auto import tqdm

from tensorflow_similarity.classification_metrics import (  # noqa
    ClassificationMetric,
    make_classification_metric,
)
from tensorflow_similarity.distances import Distance, distance_canonicalizer
from tensorflow_similarity.evaluators.evaluator import Evaluator
from tensorflow_similarity.indexer import Indexer
from tensorflow_similarity.layers import ActivationStdLoggingLayer
from tensorflow_similarity.losses import MetricLoss
from tensorflow_similarity.matchers import ClassificationMatch
from tensorflow_similarity.retrieval_metrics import RetrievalMetric
from tensorflow_similarity.search import Search
from tensorflow_similarity.stores import Store
from tensorflow_similarity.training_metrics import DistanceMetric
from tensorflow_similarity.types import (
    CalibrationResults,
    FloatTensor,
    IntTensor,
    Lookup,
    PandasDataFrame,
    Tensor,
)

# Value based on implementation from original papers.
BN_EPSILON = 1.001e-5


def get_projector(input_dim, dim=512, activation="relu", num_layers: int = 3):
    inputs = tf.keras.layers.Input((input_dim,), name="projector_input")
    x = inputs

    for i in range(num_layers - 1):
        x = tf.keras.layers.Dense(
            dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name=f"projector_layer_{i}",
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=BN_EPSILON, name=f"batch_normalization_{i}")(x)
        x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_{i}")(x)
    x = tf.keras.layers.Dense(
        dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="projector_output",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=BN_EPSILON,
        center=False,  # Page:5, Paragraph:2 of SimSiam paper
        scale=False,  # Page:5, Paragraph:2 of SimSiam paper
        name="batch_normalization_ouput",
    )(x)
    # Metric Logging layer. Monitors the std of the layer activations.
    # Degnerate solutions colapse to 0 while valid solutions will move
    # towards something like 0.0220. The actual number will depend on the layer size.
    outputs = ActivationStdLoggingLayer(name="proj_std")(x)
    projector = tf.keras.Model(inputs, outputs, name="projector")
    return projector


def get_predictor(input_dim, hidden_dim=512, activation="relu"):
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="predictor_input")
    x = inputs

    x = tf.keras.layers.Dense(
        hidden_dim,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="predictor_layer_0",
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=BN_EPSILON, name="batch_normalization_0")(x)
    x = tf.keras.layers.Activation(activation, name=f"{activation}_activation_0")(x)

    x = tf.keras.layers.Dense(
        input_dim,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="predictor_output",
    )(x)
    # Metric Logging layer. Monitors the std of the layer activations.
    # Degnerate solutions colapse to 0 while valid solutions will move
    # towards something like 0.0220. The actual number will depend on the layer size.
    outputs = ActivationStdLoggingLayer(name="pred_std")(x)
    predictor = tf.keras.Model(inputs, outputs, name="predictor")
    return predictor


def create_contrastive_model(
    *args,
    backbone: tf.keras.Model,
    projector: tf.keras.Model | None = None,
    predictor: tf.keras.Model | None = None,
    algorithm: str = "simsiam",
    **kwargs,
) -> ContrastiveModel:
    """Create a contrastive model."""
    if projector is None:
        projector = get_projector(input_dim=backbone.output_shape[-1], num_layers=2)

    input_shape = backbone.input_shape[1:]
    inputs = tf.keras.layers.Input(shape=input_shape, name="main_model_input")
    projector_features = projector(backbone(inputs))
    if algorithm == "simsiam":
        if predictor is None:
            predictor = get_predictor(input_dim=projector.output_shape[-1])
        predictor_features = predictor(projector_features)
    else:
        predictor_features = None

    outputs = [projector_features]
    if predictor_features is not None:
        outputs.append(predictor_features)

    return ContrastiveModel(
        *args,
        backbone=backbone,
        projector=projector,
        predictor=predictor,
        algorithm=algorithm,
        inputs=inputs,
        outputs=outputs,
        **kwargs,
    )


@tf.keras.utils.register_keras_serializable(package="Similarity")
class ContrastiveModel(tf.keras.Model):
    def __init__(
        self,
        *args,
        backbone: tf.keras.Model,
        projector: tf.keras.Model,
        predictor: tf.keras.Model | None = None,
        algorithm: str = "simsiam",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor

        self.algorithm = algorithm

        self._create_loss_trackers()

        self.supported_algorithms = ("simsiam", "simclr", "barlow", "vicreg")

        if self.algorithm not in self.supported_algorithms:
            raise ValueError(
                f"{self.algorithm} is not a supported algorithm."
                f"Supported algorithms are {self.supported_algorithms}."
            )

    def compile(
        self,
        optimizer: Optimizer | str | Mapping | Sequence = "rmsprop",
        loss: Loss | MetricLoss | str | Mapping | Sequence | None = None,
        metrics: Metric | DistanceMetric | str | Mapping | Sequence | None = None,  # noqa
        loss_weights: Mapping | Sequence | None = None,
        weighted_metrics: Metric | DistanceMetric | str | Mapping | Sequence | None = None,  # noqa
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        distance: Distance | str = "cosine",
        kv_store: Store | str = "memory",
        search: Search | str = "nmslib",
        evaluator: Evaluator | str = "memory",
        stat_buffer_size: int = 1000,
        **kwargs,
    ):
        """Configures the model for training.

        Args:
            optimizer: String (name of optimizer) or optimizer instance. See
              [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).

            loss: String (name of objective function), objective function, any
              `tensorflow_similarity.loss.*` instance or a `tf.keras.losses.Loss`
              instance. See the [Losses documentation](../losses.md) for a list of
              metric learning specific losses offered by TensorFlow Similarity and
              [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
              for the losses available directly in TensorFlow.

            metrics: List of metrics to be evaluated by the model during
              training and testing. Each of those can be a string, a function or a
              [tensorflow_similarity.metrics.*](../metrics.md) instance. Note that
              the metrics used for some types of metric-learning such as distance
              learning (e.g via triplet loss) have a different prototype than the
              metrics used in standard models and you can't use the
              `tf.keras.metrics` for those types of learning.

              Additionally many distance metrics are computed based of the
              [Indexer()](../indexer.md) performance. E.g Matching Top 1 accuracy.
              For technical and performance reasons, indexing data at each
              training batch to compute those is impractical so those metrics are
              computed at epoch end via the [EvalCallback](../callbacks.md)

              See [Evaluation Metrics](../eval_metrics.md) for a list of available
              metrics.

            loss_weights: Optional list or dictionary specifying scalar
              coefficients (Python floats) to weight the loss contributions of
              different model outputs. The loss value that will be minimized by
              the model will then be the *weighted sum* of all individual losses,
              weighted by the `loss_weights` coefficients.  If a list, it is
              expected to have a 1:1 mapping to the model's outputs. If a dict, it
              is expected to map output names (strings) to scalar coefficients.

            weighted_metrics: List of metrics to be evaluated and weighted by
              sample_weight or class_weight during training and testing.

            run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s
              logic will not be wrapped in a `tf.function`. Recommended to leave
              this as `None` unless your `Model` cannot be run inside a
              `tf.function`.

            steps_per_execution: Int. Defaults to 1. The number of batches to
              run during each `tf.function` call. Running multiple batches inside
              a single `tf.function` call can greatly improve performance on TPUs
              or small models with a large Python overhead.  At most, one full
              epoch will be run each execution. If a number larger than the size
              of the epoch is passed,  the execution will be truncated to the size
              of the epoch.  Note that if `steps_per_execution` is set to `N`,
              `Callback.on_batch_begin` and `Callback.on_batch_end` methods will
              only be called every `N` batches (i.e. before/after each
              `tf.function` execution).

            distance: Distance used to compute embeddings proximity.  Defaults
              to 'cosine'.

            kv_store: How to store the indexed records.  Defaults to 'memory'.

            search: Which `Search()` framework to use to perform KNN search.
              Defaults to 'nmslib'.

            evaluator: What type of `Evaluator()` to use to evaluate index
              performance. Defaults to in-memory one.

            stat_buffer_size: Size of the sliding windows buffer used to compute
              index performance. Defaults to 1000.

        Raises:
            ValueError: In case of invalid arguments for
                `optimizer`, `loss` or `metrics`.
        """
        distance_obj = distance_canonicalizer(distance)

        # init index
        self.create_index(
            distance=distance_obj,
            search=search,
            kv_store=kv_store,
            evaluator=evaluator,
            stat_buffer_size=stat_buffer_size,
        )

        # call underlying keras method
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            **kwargs,
        )

    def _create_loss_trackers(self):
        self.loss_trackers = {}
        self.loss_trackers["loss"] = tf.keras.metrics.Mean(name="loss")

        self.reg_loss_len = len(self.backbone.losses) + len(self.projector.losses)
        if self.predictor is not None:
            self.reg_loss_len += len(self.predictor.losses)

        if self.reg_loss_len:
            self.loss_trackers["contrastive_loss"] = tf.keras.metrics.Mean(name="contrastive_loss")
            self.loss_trackers["regularization_loss"] = tf.keras.metrics.Mean(name="regularization_loss")

    @property
    def metrics(self):
        # We remove the compiled loss metric because we want to manually track
        # the loss in train_step and test_step.
        base_metrics = [m for m in super().metrics if m.name != "loss"]
        loss_trackers = list(self.loss_trackers.values())
        loss_trackers.extend(base_metrics)
        return loss_trackers

    def train_step(self, data):
        view1, view2 = self._parse_views(data)

        # Forward pass through the backbone, projector, and predictor
        with tf.GradientTape() as tape:
            contrastive_loss, pred1, pred2, z1, z2 = self._forward_pass(view1, view2, training=True)
            regularization_loss = sum(self.backbone.losses)
            regularization_loss += sum(self.projector.losses)
            if self.predictor is not None:
                regularization_loss += sum(self.predictor.losses)
            combined_loss = contrastive_loss + regularization_loss

        # collect train variables from both the backbone, projector, and projector
        tvars = self.backbone.trainable_variables
        tvars = tvars + self.projector.trainable_variables

        if self.predictor is not None:
            tvars = tvars + self.predictor.trainable_variables

        # Compute gradients
        gradients = tape.gradient(combined_loss, tvars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, tvars))

        # Update metrics
        # !This are contrastive metrics with different input
        # TODO: figure out interesting metrics -- z Mae?
        # TODO: check metrics are of the right type in compile?
        # self.compiled_metrics.update_state([z1, z2], [pred2, pred1])

        # report loss manually
        self.loss_trackers["loss"].update_state(combined_loss)
        if self.reg_loss_len:
            self.loss_trackers["contrastive_loss"].update_state(contrastive_loss)
            self.loss_trackers["regularization_loss"].update_state(regularization_loss)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        view1, view2 = self._parse_views(data)

        contrastive_loss, pred1, pred2, z1, z2 = self._forward_pass(view1, view2, training=False)
        regularization_loss = sum(self.backbone.losses)
        regularization_loss += sum(self.projector.losses)
        if self.predictor is not None:
            regularization_loss += sum(self.predictor.losses)
        combined_loss = contrastive_loss + regularization_loss

        # Update metrics
        # !This are contrastive metrics with different input
        # TODO: figure out interesting metrics -- z Mae?
        # TODO: check metrics are of the right type in compile?
        # self.compiled_metrics.update_state([z1, z2], [pred2, pred1])

        # report loss manually
        self.loss_trackers["loss"].update_state(combined_loss)
        if self.reg_loss_len:
            self.loss_trackers["contrastive_loss"].update_state(contrastive_loss)
            self.loss_trackers["regularization_loss"].update_state(regularization_loss)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def _forward_pass(self, view1, view2, training):
        h1 = self.backbone(view1, training=training)
        h2 = self.backbone(view2, training=training)

        z1 = self.projector(h1, training=training)
        z2 = self.projector(h2, training=training)

        if self.algorithm == "simsiam":
            p1 = self.predictor(z1, training=training)
            p2 = self.predictor(z2, training=training)
            l1 = self.compiled_loss(tf.stop_gradient(z1), p2)
            l2 = self.compiled_loss(tf.stop_gradient(z2), p1)
            loss = l1 + l2
            pred1, pred2 = p1, p2
        elif self.algorithm in ("simclr", "barlow", "vicreg"):
            loss = self.compiled_loss(z1, z2)
            pred1, pred2 = z1, z2

        return loss, pred1, pred2, z1, z2

    def _parse_views(self, data: Sequence[FloatTensor]) -> tuple[FloatTensor, FloatTensor]:
        if len(data) == 2:
            view1, view2 = data
        else:
            view1 = view2 = data[0]

        return view1, view2

    # fix TF 2.x < 2.7 bugs when using generator
    # todo(ovallis): link to original issue.
    def call(self, inputs, training=None, mask=None):
        return inputs

    def summary(self):
        cprint("[Backbone]", "green")
        self.backbone.summary()
        cprint("\n[Projector]", "magenta")
        self.projector.summary()
        if self.predictor is not None:
            cprint("\n[Predictor]", "magenta")
            self.projector.summary()

    def predict(
        self,
        x: FloatTensor,
        batch_size: int | None = None,
        verbose: int = 0,
        steps: int | None = None,
        callbacks: tf.keras.callbacks.Callback | None = None,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
    ) -> FloatTensor:
        """Generates output predictions for the input samples.

        The predict returns the L2 normalized output of the projector.

        This can be used for indexing and querying examples, and is used by the
        EvalCallback.
        """
        x = self.backbone.predict(
            x,
            batch_size,
            verbose,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing,
        )
        x = self.projector.predict(
            x,
            batch_size,
            verbose,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing,
        )

        output: FloatTensor = tf.math.l2_normalize(x, axis=1)

        return output

    def create_index(
        self,
        distance: Distance | str = "cosine",
        search: Search | str = "nmslib",
        kv_store: Store | str = "memory",
        evaluator: Evaluator | str = "memory",
        stat_buffer_size: int = 1000,
    ) -> None:
        """Create the model index to make embeddings searchable via KNN.

        This method is normally called as part of `SimilarityModel.compile()`.
        However, this method is provided if users want to define a custom index
        outside of the `compile()` method.

        NOTE: This method sets `SimilarityModel._index` and will replace any
        existing index.

        Args:
            distance: Distance used to compute embeddings proximity. Defaults to
            'auto'.

            kv_store: How to store the indexed records.  Defaults to 'memory'.

            search: Which `Search()` framework to use to perform KNN search.
            Defaults to 'nmslib'.

            evaluator: What type of `Evaluator()` to use to evaluate index
            performance. Defaults to in-memory one.

            stat_buffer_size: Size of the sliding windows buffer used to compute
            index performance. Defaults to 1000.

        Raises:
            ValueError: Invalid search framework or key value store.
        """
        self._index = Indexer(
            embedding_size=self.projector.output_shape[-1],
            distance=distance,
            search=search,
            kv_store=kv_store,
            evaluator=evaluator,
            embedding_output=None,
            stat_buffer_size=stat_buffer_size,
        )

    def index(
        self,
        x: Tensor,
        y: IntTensor | None = None,
        data: Tensor | None = None,
        build: bool = True,
        verbose: int = 1,
    ):
        """Index data.

        Args:
            x: Samples to index.

            y: class ids associated with the data if any. Defaults to None.

            store_data: store the data associated with the samples in the key
            value store. Defaults to True.

            build: Rebuild the index after indexing. This is needed to make the
            new samples searchable. Set it to false to save processing time
            when calling indexing repeatidly without the need to search between
            the indexing requests. Defaults to True.

            verbose: Output indexing progress info. Defaults to 1.
        """

        if not self._index:
            raise Exception("You need to compile the model with a valid" "distance to be able to use the indexing")
        if verbose:
            print("[Indexing %d points]" % len(x))
            print("|-Computing embeddings")
        predictions = self.predict(x)

        self._index.batch_add(
            predictions=predictions,
            labels=y,
            data=data,
            build=build,
            verbose=verbose,
        )

    def index_single(
        self,
        x: Tensor,
        y: IntTensor | None = None,
        data: Tensor | None = None,
        build: bool = True,
        verbose: int = 1,
    ):
        """Index data.

        Args:
            x: Sample to index.

            y: class id associated with the data if any. Defaults to None.

            data: store the data associated with the samples in the key
            value store. Defaults to None.

            build: Rebuild the index after indexing. This is needed to make the
            new samples searchable. Set it to false to save processing time
            when calling indexing repeatidly without the need to search between
            the indexing requests. Defaults to True.

            verbose: Output indexing progress info. Defaults to 1.
        """

        if not self._index:
            raise Exception("You need to compile the model with a valid" "distance to be able to use the indexing")
        if verbose:
            print("[Indexing 1 point]")
            print("|-Computing embeddings")

        x = tf.expand_dims(x, axis=0)
        prediction = self.predict(x)
        self._index.add(
            prediction=prediction,
            label=y,
            data=data,
            build=build,
            verbose=verbose,
        )

    def lookup(self, x: Tensor, k: int = 5, verbose: int = 1) -> list[list[Lookup]]:
        """Find the k closest matches in the index for a set of samples.

        Args:
            x: Samples to match.

            k: Number of nearest neighboors to lookup. Defaults to 5.

            verbose: display progress. Default to 1.

        Returns
            list of list of k nearest neighboors:
            list[list[Lookup]]
        """
        predictions = self.predict(x)
        return self._index.batch_lookup(predictions=predictions, k=k, verbose=verbose)

    def single_lookup(self, x: Tensor, k: int = 5) -> list[Lookup]:
        """Find the k closest matches in the index for a given sample.

        Args:
            x: Sample to match.

            k: Number of nearest neighboors to lookup. Defaults to 5.

        Returns
            list of the k nearest neigboors info:
            list[Lookup]
        """
        x = tf.expand_dims(x, axis=0)
        prediction = self.predict(x)
        return self._index.single_lookup(prediction=prediction, k=k)

    def index_summary(self):
        "Display index info summary."
        self._index.print_stats()

    def calibrate(
        self,
        x: FloatTensor,
        y: IntTensor,
        thresholds_targets: MutableMapping[str, float] = {},
        k: int = 1,
        calibration_metric: str | ClassificationMetric = "f1",
        matcher: str | ClassificationMatch = "match_nearest",
        extra_metrics: MutableSequence[str | ClassificationMetric] = [
            "precision",
            "recall",
        ],  # noqa
        rounding: int = 2,
        verbose: int = 1,
    ) -> CalibrationResults:
        """Calibrate model thresholds using a test dataset.

        TODO: more detailed explaination.

        Args:

            x: examples to use for the calibration.

            y: labels associated with the calibration examples.

            thresholds_targets: Dict of performance targets to (if possible)
            meet with respect to the `calibration_metric`.

            calibration_metric:
            [ClassificationMetric()](classification_metrics/overview.md) used
            to evaluate the performance of the index.

            k: How many neighboors to use during the calibration.
            Defaults to 1.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold. Defaults to
            'match_nearest'.

            extra_metrics: List of additional
            `tf.similarity.classification_metrics.ClassificationMetric()`
            to compute and report. Defaults to ['precision', 'recall'].


            rounding: Metric rounding. Default to 2 digits.

            verbose: Be verbose and display calibration results.
            Defaults to 1.

        Returns:
            CalibrationResults containing the thresholds and cutpoints Dicts.
        """

        # predict
        predictions = self.predict(x)

        # calibrate
        return self._index.calibrate(
            predictions=predictions,
            target_labels=y,
            thresholds_targets=thresholds_targets,
            k=k,
            calibration_metric=calibration_metric,
            matcher=matcher,
            extra_metrics=extra_metrics,
            rounding=rounding,
            verbose=verbose,
        )

    def match(
        self,
        x: FloatTensor,
        cutpoint="optimal",
        no_match_label=-1,
        k=1,
        matcher: str | ClassificationMatch = "match_nearest",
        verbose=0,
    ):
        """Match a set of examples against the calibrated index

        For the match function to work, the index must be calibrated using
        calibrate().

        Args:
            x: Batch of examples to be matched against the index.

            cutpoint: Which calibration threshold to use.
            Defaults to 'optimal' which is the optimal F1 threshold computed
            using calibrate().

            no_match_label: Which label value to assign when there is no
            match. Defaults to -1.

            k: How many neighboors to use during the calibration.
            Defaults to 1.

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            verbose. Be verbose. Defaults to 0.

        Returns:
            List of class ids that matches for each supplied example

        Notes:
            This function matches all the cutpoints at once internally as there
            is little performance downside to do so and allows to do the
            evaluation in a single go.

        """
        # basic checks
        if not self._index.is_calibrated:
            raise ValueError("Uncalibrated model: run model.calibration()")

        # get predictions
        predictions = self.predict(x)

        # matching
        matches = self._index.match(
            predictions,
            no_match_label=no_match_label,
            k=k,
            matcher=matcher,
            verbose=verbose,
        )

        # select which matches to return
        if cutpoint == "all":  # returns all the cutpoints for eval purpose.
            return matches
        else:  # normal match behavior - returns a specific cut point
            return matches[cutpoint]

    def evaluate_retrieval(
        self,
        x: Tensor,
        y: IntTensor,
        retrieval_metrics: Sequence[RetrievalMetric],  # noqa
        verbose: int = 1,
    ) -> dict[str, np.ndarray]:
        """Evaluate the quality of the index against a test dataset.

        Args:
            x: Examples to be matched against the index.

            y: Label associated with the examples supplied.

            retrieval_metrics: List of
            [RetrievalMetric()](retrieval_metrics/overview.md) to compute.

            verbose (int, optional): Display results if set to 1 otherwise
            results are returned silently. Defaults to 1.

        Returns:
            Dictionary of metric results where keys are the metric names and
            values are the metrics values.

        Raises:
            IndexError: Index must contain embeddings but is currently empty.
        """
        if self._index.size() == 0:
            raise IndexError("Index must contain embeddings but is " "currently empty. Have you run model.index()?")

        # get embeddings
        if verbose:
            print("|-Computing embeddings")
        predictions = self.predict(x)

        if verbose:
            print("|-Computing retrieval metrics")

        results = self._index.evaluate_retrieval(
            predictions=predictions,
            target_labels=y,
            retrieval_metrics=retrieval_metrics,
            verbose=verbose,
        )

        if verbose:
            table = zip(results.keys(), results.values())
            headers = ["metric", "Value"]
            print("\n [Summary]\n")
            print(tabulate(table, headers=headers))

        return results

    def evaluate_classification(
        self,
        x: Tensor,
        y: IntTensor,
        k: int = 1,
        extra_metrics: MutableSequence[str | ClassificationMetric] = [
            "precision",
            "recall",
        ],  # noqa
        matcher: str | ClassificationMatch = "match_nearest",
        verbose: int = 1,
    ) -> defaultdict[str, dict[str, str | np.ndarray]]:
        """Evaluate model classification matching on a given evaluation dataset.

        Args:
            x: Examples to be matched against the index.

            y: Label associated with the examples supplied.

            k: How many neighbors to use to perform the evaluation.
            Defaults to 1.

            extra_metrics: List of additional
            `tf.similarity.classification_metrics.ClassificationMetric()` to
            compute and report. Defaults to ['precision', 'recall'].

            matcher: {'match_nearest', 'match_majority_vote'} or
            ClassificationMatch object. Defines the classification matching,
            e.g., match_nearest will count a True Positive if the query_label
            is equal to the label of the nearest neighbor and the distance is
            less than or equal to the distance threshold.

            verbose (int, optional): Display results if set to 1 otherwise
            results are returned silently. Defaults to 1.

        Returns:
            Dictionary of (distance_metrics.md)[evaluation metrics]

        Raises:
            IndexError: Index must contain embeddings but is currently empty.
            ValueError: Uncalibrated model: run model.calibration()")
        """
        # There is some code duplication in this function but that is the best
        # solution to keep the end-user API clean and doing inferences once.
        if self._index.size() == 0:
            raise IndexError("Index must contain embeddings but is " "currently empty. Have you run model.index()?")

        if not self._index.is_calibrated:
            raise ValueError("Uncalibrated model: run model.calibration()")

        cal_metric = self._index.get_calibration_metric()

        # get embeddings
        if verbose:
            print("|-Computing embeddings")
        predictions = self.predict(x)

        results: defaultdict[str, dict[str, str | np.ndarray]] = defaultdict(dict)

        if verbose:
            pb = tqdm(total=len(self._index.cutpoints), desc="Evaluating cutpoints")

        for cp_name, cp_data in self._index.cutpoints.items():
            # create a metric that match at the requested k and threshold
            distance_threshold = float(cp_data["distance"])
            metric = make_classification_metric(cal_metric.name)
            metrics = copy(extra_metrics)
            metrics.append(metric)

            res: dict[str, str | np.ndarray] = {}
            res.update(
                self._index.evaluate_classification(
                    predictions,
                    y,
                    [distance_threshold],
                    metrics=metrics,
                    matcher=matcher,
                    k=k,
                )
            )
            res["distance"] = tf.constant([distance_threshold])
            res["name"] = cp_name
            results[cp_name] = res
            if verbose:
                pb.update()

        if verbose:
            pb.close()

        if verbose:
            headers = ["name", cal_metric.name]
            for i in results["optimal"].keys():
                if i not in headers:
                    headers.append(str(i))
            rows = []
            for data in results.values():
                rows.append([data[v] for v in headers])
            print("\n [Summary]\n")
            print(tabulate(rows, headers=headers))

        return results

    def reset_index(self):
        "Reinitialize the index"
        self._index.reset()

    def index_size(self) -> int:
        "Return the index size"
        return self._index.size()

    def load_index(self, filepath: str):
        """Load Index data from a checkpoint and initialize underlying
        structure with the reloaded data.

        Args:
            path: Directory where the checkpoint is located.
            verbose: Be verbose. Defaults to 1.
        """

        index_path = Path(filepath) / "index"
        self._index = Indexer.load(index_path)

    def save_index(self, filepath, compression=True):
        """Save the index to disk

        Args:
            path: directory where to save the index
            compression: Store index data compressed. Defaults to True.
        """
        index_path = Path(filepath) / "index"
        self._index.save(index_path, compression=compression)

    def save(
        self,
        filepath: str | Path,
        save_index: bool = True,
        compression: bool = True,
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: str | None = None,
        signatures: Callable | Mapping[str, Callable] | None = None,
        options: tf.saved_model.SaveOptions | None = None,
        save_traces: bool = True,
    ) -> None:
        """Save Constrative model backbone, projector, and predictor

        Args:
            filepath: where to save the model.
            save_index: Save the index content. Defaults to True.
            compression: Compress index data. Defaults to True.
            overwrite: Overwrite previous model. Defaults to True.
            include_optimizer: Save optimizer state. Defaults to True.
            save_format: Either 'tf' or 'h5', indicating whether to save the
              model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in
              TF 2.X, and 'h5' in TF 1.X.
            signatures: Signatures to save with the SavedModel. Applicable to
              the 'tf' format only. Please see the signatures argument in
              tf.saved_model.save for details.
            options: A `tf.saved_model.SaveOptions` to save with the model.
              Defaults to None.
            save_traces (optional): When enabled, the SavedModel will store the
              function traces for each layer. This can be disabled, so that only
              the configs of each layer are stored.  Defaults to True. Disabling
              this will decrease serialization time and reduce file size, but it
              requires that all custom layers/models implement a get_config()
              method.
        """
        super().save(
            filepath,
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

        if hasattr(self, "_index") and self._index and save_index:
            self.save_index(filepath, compression=compression)
        else:
            msg = "The index was not saved with the model."
            if not hasattr(self, "_index"):
                msg = msg + (
                    "The model does not currently have an index. To use indexing "
                    "you must call either model.compile() or model.create_index() "
                    "and set a valid Distance."
                )

            if not save_index:
                msg = msg + " The save_index param is set to False."

            print(msg)

    def to_data_frame(self, num_items: int = 0) -> PandasDataFrame:
        """Export data as pandas dataframe

        Args:
            num_items (int, optional): Num items to export to the dataframe.
            Defaults to 0 (unlimited).

        Returns:
            pd.DataFrame: a pandas dataframe.
        """
        return self._index.to_data_frame(num_items=num_items)

    def get_config(self) -> dict[str, Any]:
        config = {
            "backbone": self.backbone,
            "projector": self.projector,
            "predictor": self.predictor,
            "algorithm": self.algorithm,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "layers" in config:
            del config["layers"]
        if "input_layers" in config:
            del config["input_layers"]
        if "output_layers" in config:
            del config["output_layers"]
        return create_contrastive_model(**config)
