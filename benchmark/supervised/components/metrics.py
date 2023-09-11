from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.retrieval_metrics import (
    MapAtK,
    PrecisionAtK,
    RecallAtK,
    RetrievalMetric,
)
from tensorflow_similarity.types import FloatTensor, IntTensor


def make_eval_metrics(ecfg: Mapping[str, Any], class_counts: Mapping[int, int]) -> list[RetrievalMetric]:
    metrics = []
    for metric_id, params in ecfg.items():
        if metric_id == "recall_at_k":
            for k in params["k"]:
                metrics.append(
                    RecallAtK(
                        k=k,
                        average=params.get("average", "micro"),
                        drop_closest_lookup=True,
                    )
                )
        elif metric_id == "precision_at_k":
            for k in params["k"]:
                metrics.append(
                    PrecisionAtK(
                        k=k,
                        average=params.get("average", "micro"),
                        drop_closest_lookup=True,
                    )
                )
        elif metric_id == "map_at_r":
            max_class_count = max(class_counts.values())
            metrics.append(
                MapAtK(
                    r=class_counts,
                    clip_at_r=True,
                    k=max_class_count,
                    drop_closest_lookup=True,
                    name="map@R",
                )
            )
        elif metric_id == "r_precision":
            max_class_count = max(class_counts.values())
            metrics.append(
                PrecisionAtK(
                    r=class_counts,
                    clip_at_r=True,
                    k=max_class_count,
                    drop_closest_lookup=True,
                    name="R_Precision",
                )
            )
        else:
            raise ValueError(f"Unknown metric name: {metric_id}")

    return metrics


def make_eval_callback(
    val_x: Sequence[FloatTensor],
    val_y: Sequence[IntTensor],
    aug_fns: tuple[Any, ...],
    num_queries: int,
    num_targets: int,
) -> EvalCallback:
    # Setup EvalCallback by splitting the test data into targets and queries.
    num_examples = len(val_x)
    num_queries = min(num_queries, num_examples // 2)
    num_targets = min(num_examples - num_queries, num_targets)
    print(f"Make eval callback with {num_queries} queries and {num_targets} targets")

    queries_x = val_x[:num_queries]

    for aug_fn in aug_fns:
        queries_x = tf.map_fn(aug_fn, queries_x)
    queries_y = val_y[:num_queries]

    targets_x = val_x[num_queries : num_queries + num_targets]
    for aug_fn in aug_fns:
        targets_x = tf.map_fn(aug_fn, targets_x)
    targets_y = val_y[num_queries : num_queries + num_targets]

    unique, counts = np.unique(targets_y, return_counts=True)
    class_counts = {k: v for k, v in zip(unique, counts)}
    retrieval_metrics = make_eval_metrics({"map_at_r": {}, "r_precision": {}}, class_counts)

    return EvalCallback(
        queries_x,
        queries_y,
        targets_x,
        targets_y,
        metrics=["binary_accuracy"],
        k=max(class_counts.values()),
        retrieval_metrics=retrieval_metrics,
    )
