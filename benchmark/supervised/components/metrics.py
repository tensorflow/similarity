from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.retrieval_metrics import MapAtK, PrecisionAtK, RecallAtK


def make_eval_metrics(tconf, econf, class_counts):
    metrics = []
    for metric_name, params in econf.items():
        if metric_name == "recall_at_k":
            for k in params["k"]:
                metrics.append(
                    RecallAtK(
                        k=k,
                        average=params.get("average", "micro"),
                        drop_closest_lookup=True,
                    )
                )
        elif metric_name == "precision_at_k":
            for k in params["k"]:
                metrics.append(
                    PrecisionAtK(
                        k=k,
                        average=params.get("average", "micro"),
                        drop_closest_lookup=True,
                    )
                )
        elif metric_name == "map_at_r":
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
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")

    return metrics


def make_eval_callback(val_ds, num_queries, num_targets):
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
    return EvalCallback(
        queries_x,
        queries_y,
        targets_x,
        targets_y,
        metrics=["binary_accuracy", "precision"],
        k=1,
    )
