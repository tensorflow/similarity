import math

import numpy as np
import tensorflow as tf

from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.classification_metrics import Recall
from tensorflow_similarity.evaluators import MemoryEvaluator
from tensorflow_similarity.models import SimilarityModel


def test_eval_init_defaults():
    queries = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    target_labels = tf.constant([1, 2])

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
    )

    assert tf.math.reduce_all(callback.queries_known == queries)
    assert tf.math.reduce_all(callback.query_labels_known == query_labels)
    assert tf.math.reduce_all(callback.targets == targets)
    assert tf.math.reduce_all(callback.target_labels == target_labels)
    assert callback.distance == "cosine"
    assert isinstance(callback.evaluator, MemoryEvaluator)
    assert {"binary_accuracy"} == set([m.name for m in callback.classification_metrics])
    assert callback.k == 1
    assert tf.math.reduce_all(callback.distance_thresholds == tf.constant([math.inf]))
    assert callback.matcher == "match_nearest"
    assert callback.tb_writer is None


def test_eval_init(tmp_path):
    queries = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    target_labels = tf.constant([1, 2])
    distance = "l2"
    metrics = [Recall()]
    k = 11
    matcher = "majority_vote"
    distance_thresholds = tf.constant([0.1, 0.2, 0.3])

    log_dir = tmp_path / "sec/"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        distance=distance,
        metrics=metrics,
        k=k,
        matcher=matcher,
        distance_thresholds=distance_thresholds,
        tb_logdir=log_dir,
    )

    assert tf.math.reduce_all(callback.queries_known == queries)
    assert tf.math.reduce_all(callback.query_labels_known == query_labels)
    assert tf.math.reduce_all(callback.targets == targets)
    assert tf.math.reduce_all(callback.target_labels == target_labels)
    assert callback.distance == "l2"
    assert isinstance(callback.evaluator, MemoryEvaluator)
    assert {"recall"} == set([m.name for m in callback.classification_metrics])
    assert callback.k == 11
    assert tf.math.reduce_all(callback.distance_thresholds == distance_thresholds)
    assert callback.matcher == "majority_vote"
    assert isinstance(callback.tb_writer, tf.summary.SummaryWriter)


def test_eval_callback(tmp_path):
    queries = tf.constant([[1.0, 2.0], [1.0, 2.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [1.0, 2.0]])
    target_labels = tf.constant([1, 2])

    log_dir = tmp_path / "sec/"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        tb_logdir=str(log_dir),
        metrics=["binary_accuracy", "f1score"],
    )

    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile("adam", loss="mse", distance="cosine")
    callback.model = model

    # call the only callback method implemented
    logs = {}
    callback.on_epoch_end(0, logs)

    metric_values = np.array(list(logs.values()))
    assert {"binary_accuracy", "f1score"} == logs.keys()
    np.testing.assert_allclose(np.array([0.5, 0.666667]), metric_values, rtol=1e-5, atol=0)


def test_split_val_init_defaults():
    queries = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    target_labels = tf.constant([1, 2])
    known_classes = tf.constant([1])
    q_known = tf.constant([[1.0, 2.0]])
    q_label_known = tf.constant([1], dtype="int32")
    q_unknown = tf.constant([[3.0, 4.0]])
    q_label_unknown = tf.constant([2], dtype="int32")

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        known_classes=known_classes,
    )

    assert tf.math.reduce_all(callback.targets == targets)
    assert tf.math.reduce_all(callback.target_labels == target_labels)
    assert callback.distance == "cosine"
    assert isinstance(callback.evaluator, MemoryEvaluator)
    assert {"binary_accuracy"} == set([m.name for m in callback.classification_metrics])
    assert callback.k == 1
    assert tf.math.reduce_all(callback.distance_thresholds == tf.constant([math.inf]))
    assert callback.matcher == "match_nearest"
    assert callback.tb_writer is None

    assert tf.math.reduce_all(callback.queries_known == q_known)
    assert tf.math.reduce_all(callback.query_labels_known == q_label_known)
    assert tf.math.reduce_all(callback.queries_unknown == q_unknown)
    assert tf.math.reduce_all(callback.query_labels_unknown == q_label_unknown)


def test_split_val_init(tmp_path):
    queries = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    target_labels = tf.constant([1, 2])
    known_classes = tf.constant([1])
    distance = "l2"
    metrics = [Recall()]
    k = 11
    matcher = "majority_vote"
    distance_thresholds = tf.constant([0.1, 0.2, 0.3])
    q_known = tf.constant([[1.0, 2.0]])
    q_label_known = tf.constant([1], dtype="int32")
    q_unknown = tf.constant([[3.0, 4.0]])
    q_label_unknown = tf.constant([2], dtype="int32")

    log_dir = tmp_path / "sec/"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        known_classes=known_classes,
        distance=distance,
        metrics=metrics,
        k=k,
        matcher=matcher,
        distance_thresholds=distance_thresholds,
        tb_logdir=log_dir,
    )

    assert tf.math.reduce_all(callback.targets == targets)
    assert tf.math.reduce_all(callback.target_labels == target_labels)
    assert callback.distance == "l2"
    assert isinstance(callback.evaluator, MemoryEvaluator)
    assert {"recall"} == set([m.name for m in callback.classification_metrics])
    assert callback.k == 11
    assert tf.math.reduce_all(callback.distance_thresholds == distance_thresholds)
    assert callback.matcher == "majority_vote"
    assert isinstance(callback.tb_writer, tf.summary.SummaryWriter)

    assert tf.math.reduce_all(callback.queries_known == q_known)
    assert tf.math.reduce_all(callback.query_labels_known == q_label_known)
    assert tf.math.reduce_all(callback.queries_unknown == q_unknown)
    assert tf.math.reduce_all(callback.query_labels_unknown == q_label_unknown)


def test_split_val_loss_callback(tmp_path):
    queries = tf.constant([[1.0, 2.0], [1.0, 2.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [1.0, 2.0]])
    target_labels = tf.constant([1, 2])
    known_classes = tf.constant([1])

    log_dir = tmp_path / "sec/"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        known_classes=known_classes,
        tb_logdir=str(log_dir),
    )

    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile("adam", loss="mse", distance="cosine")
    callback.model = model

    # call the only callback method implemented
    callback.on_epoch_end(0, {})


def test_split_val_eval_init(tmp_path):
    queries = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    target_labels = tf.constant([1, 2])
    known_classes = tf.constant([1])
    distance = "l2"
    metrics = [Recall()]
    k = 11
    matcher = "majority_vote"
    distance_thresholds = tf.constant([0.1, 0.2, 0.3])
    q_known = tf.constant([[1.0, 2.0]])
    q_label_known = tf.constant([1], dtype="int32")
    q_unknown = tf.constant([[3.0, 4.0]])
    q_label_unknown = tf.constant([2], dtype="int32")

    log_dir = tmp_path / "sec/"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        known_classes=known_classes,
        distance=distance,
        metrics=metrics,
        k=k,
        matcher=matcher,
        distance_thresholds=distance_thresholds,
        tb_logdir=log_dir,
    )

    assert tf.math.reduce_all(callback.targets == targets)
    assert tf.math.reduce_all(callback.target_labels == target_labels)
    assert callback.distance == "l2"
    assert isinstance(callback.evaluator, MemoryEvaluator)
    assert {"recall"} == set([m.name for m in callback.classification_metrics])
    assert callback.k == 11
    assert tf.math.reduce_all(callback.distance_thresholds == distance_thresholds)
    assert callback.matcher == "majority_vote"
    assert isinstance(callback.tb_writer, tf.summary.SummaryWriter)

    assert tf.math.reduce_all(callback.queries_known == q_known)
    assert tf.math.reduce_all(callback.query_labels_known == q_label_known)
    assert tf.math.reduce_all(callback.queries_unknown == q_unknown)
    assert tf.math.reduce_all(callback.query_labels_unknown == q_label_unknown)


def test_split_val_eval_callback(tmp_path):
    queries = tf.constant([[1.0, 2.0], [1.0, 2.0]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1.0, 2.0], [1.0, 2.0]])
    target_labels = tf.constant([1, 2])
    known_classes = tf.constant([1])

    log_dir = tmp_path / "sec/"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(
        queries=queries,
        query_labels=query_labels,
        targets=targets,
        target_labels=target_labels,
        known_classes=known_classes,
        tb_logdir=str(log_dir),
    )

    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile("adam", loss="mse", distance="cosine")
    callback.model = model

    # call the only callback method implemented
    callback.on_epoch_end(0, {})
