from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.retrieval_metrics import (
    MapAtK,
    PrecisionAtK,
    RecallAtK,
    RetrievalMetric,
)
from tensorflow_similarity.samplers import MultiShotMemorySampler


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


def make_eval_callback(val_ds: MultiShotMemorySampler, num_queries: int, num_targets: int) -> EvalCallback:
    # Setup EvalCallback by splitting the test data into targets and queries.
    num_queries = min(num_queries, val_ds.num_examples // 2)
    num_targets = min(val_ds.num_examples - num_queries, num_targets)
    print(f"Make eval callback with {num_queries} queries and {num_targets} targets")

    queries_x, queries_y = val_ds.get_slice(0, num_queries)
    queries_x, queries_y = val_ds.augmenter(
        queries_x, queries_y, val_ds.num_augmentations_per_example, val_ds.is_warmup
    )
    targets_x, targets_y = val_ds.get_slice(num_queries, num_targets)
    targets_x, targets_y = val_ds.augmenter(
        targets_x, targets_y, val_ds.num_augmentations_per_example, val_ds.is_warmup
    )

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
