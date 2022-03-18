import os
from posixpath import split

import tensorflow as tf
import numpy as np
import argparse
import json

from termcolor import cprint
from benchmark import load_tfrecord_unbatched_dataset
from tensorflow_similarity.losses.circle_loss import CircleLoss
from tensorflow_similarity.losses import (
    TripletLoss,
    CircleLoss,
    PNLoss,
    MultiSimilarityLoss,
)
from tensorflow_similarity.models.similarity_model import SimilarityModel
from tensorflow_similarity.visualization import viz_neigbors_imgs, confusion_matrix
from tensorflow_similarity.retrieval_metrics.recall_at_k import RecallAtK


# Helpers


def load_data(version, dataset_name):
    ds_index = load_tfrecord_unbatched_dataset(version, dataset_name, "index")
    cprint("|---- loaded index", "green")
    ds_seen_queries = load_tfrecord_unbatched_dataset(
        version, dataset_name, "seen_queries"
    )
    cprint("|---- loaded seen queries", "green")
    ds_unseen_queries = load_tfrecord_unbatched_dataset(
        version, dataset_name, "unseen_queries"
    )
    cprint("|---- loaded unseen queries", "green")

    return ds_index, ds_seen_queries, ds_unseen_queries


def load_model(version, dataset_name, model_loss):
    custom_objects = {"Similarity>CircleLoss": CircleLoss, "Similarity>PNLoss": PNLoss}
    with tf.keras.utils.custom_object_scope(custom_objects):
        path = f"../../models/{version}/{dataset_name}_{model_loss}"
        full_path = os.path.join(os.path.dirname(__file__), path)

        model = tf.keras.models.load_model(full_path)
        sim_model = SimilarityModel.from_config(model.get_config())
        cprint(f"|---- loaded model", "green")
        return sim_model


def split_dataset(ds):
    return tf.data.Dataset.get_single_element(ds.batch(len(ds)))


# Refactor this into __init__.py in benchmark
def make_loss(distance, params):
    if params["loss"] == "triplet_loss":
        return TripletLoss(
            distance=distance, negative_mining_strategy=params["negative_mining"]
        )
    elif params["loss"] == "pn_loss":
        return PNLoss(
            distance=distance, negative_mining_strategy=params["negative_mining"]
        )
    elif params["loss"] == "circle_loss":
        return CircleLoss(
            distance=distance, margin=params["margin"], gamma=params["gamma"]
        )
    elif params["loss"] == "multi_similarity":
        return MultiSimilarityLoss(
            distance=distance,
            alpha=params["alpha"],
            beta=params["beta"],
            epsilon=params["epsilon"],
            lmda=params["lambda"],
        )
    else:
        raise ValueError("Unknown loss name", params["loss"])


def evaluate(
    version,
    dataset_name,
    lparams,
    distance,
    ds_index_x,
    ds_index_y,
    ds_seen_queries_x,
    ds_seen_queries_y,
    ds_unseen_queries_x,
    ds_unseen_queries_y,
):
    cprint("\n|- loading model", "blue")

    model = load_model(version, dataset_name, lparams["name"])
    loss = make_loss(distance, lparams)
    optim = tf.keras.optimizers.Adam(lparams["lr"])
    model.compile(optimizer=optim, loss=loss)

    cprint("\n|- doing model indexing", "blue")

    model.reset_index()
    model.index(ds_index_x, ds_index_y, data=ds_index_x)

    cprint("\n|- doing model calibration", "blue")
    cprint("|---- seen queries", "green")
    calibration_seen_queries = model.calibrate(
        ds_seen_queries_x,
        ds_seen_queries_y,
        calibration_metric="f1",
        matcher="match_nearest",
        extra_metrics=["precision", "recall", "binary_accuracy"],
        verbose=1,
    )
    cprint("\n|---- unseen queries", "green")
    calibration_unseen_queries = model.calibrate(
        ds_seen_queries_x,
        ds_seen_queries_y,
        calibration_metric="f1",
        matcher="match_nearest",
        extra_metrics=["precision", "recall", "binary_accuracy"],
        verbose=1,
    )
    

    # Uncomment for visualization - Proceed with caution if you run this outside of an iPython notebook
    # You will have to close the window 2 * num_samples times

    # cprint("\n|- visualization", "blue")

    # num_neighbors = 5
    # num_samples = 1

    # cprint("|---- seen queries", "green")
    # for i in range(num_samples):
    #     nns = model.single_lookup(ds_seen_queries_x[i], k=num_neighbors)
    #     viz_neigbors_imgs(ds_seen_queries_x[i], ds_seen_queries_y[i], nns)

    # cprint("\n|---- unseen queries", "green")
    # for i in range(num_samples):
    #     nns = model.single_lookup(ds_unseen_queries_x[i], k=num_neighbors)
    #     viz_neigbors_imgs(ds_unseen_queries_x[i], ds_unseen_queries_y[i], nns)

    cprint("\n|- evaluating model retrieval metrics R@1, R@2, R@4, R@8", "blue")

    cprint("|---- seen queries", "green")
    retrieval_seen_queries = model.evaluate_retrieval(
        ds_seen_queries_x,
        ds_seen_queries_y,
        retrieval_metrics=[
            RecallAtK(k=1),
            RecallAtK(k=2),
            RecallAtK(k=4),
            RecallAtK(k=8),
        ],
    )

    cprint("\n|---- unseen queries", "green")
    retrieval_unseen_queries = model.evaluate_retrieval(
        ds_unseen_queries_x,
        ds_unseen_queries_y,
        retrieval_metrics=[
            RecallAtK(k=1),
            RecallAtK(k=2),
            RecallAtK(k=4),
            RecallAtK(k=8),
        ],
    )

    cprint(f"\n {lparams['name']} evaluation results:", "green")

    results = f"""
    Seen queries:
    \tR@1: {retrieval_seen_queries['recall@1']}
    \tR@2: {retrieval_seen_queries['recall@2']}
    \tR@4: {retrieval_seen_queries['recall@4']}
    \tR@8: {retrieval_seen_queries['recall@8']}

    Unseen queries:
    \tR@1: {retrieval_unseen_queries['recall@1']}
    \tR@2: {retrieval_unseen_queries['recall@2']}
    \tR@4: {retrieval_unseen_queries['recall@4']}
    \tR@8: {retrieval_unseen_queries['recall@8']}
    """

    cprint(results, "green")

    cprint("\nSaving Results...", "yellow")

    save_path = (
        f"../../models/{version}/{dataset_name}_{lparams['name']}/evaluation.json"
    )
    full_save_path = os.path.join(os.path.dirname(__file__), save_path)

    mode = list(calibration_seen_queries.cutpoints.keys())[0]
    config_json = {
        "loss": lparams['name'],
        "dataset_name": dataset_name,
        "seen_queries": {
            "calibration": {
                "method": mode,
                "value": str(calibration_seen_queries.cutpoints[mode]["value"]),
                "distance": str(calibration_seen_queries.cutpoints[mode]["distance"]),
                "precision": str(calibration_seen_queries.cutpoints[mode]["precision"]),
                "recall": str(calibration_seen_queries.cutpoints[mode]["recall"]),
                "binary_accuracy": str(calibration_seen_queries.cutpoints[mode]["binary_accuracy"]),
                "f1": str(calibration_seen_queries.cutpoints[mode]["f1"])
            },
            "paper": {
                "R@1": str(retrieval_seen_queries["recall@1"]),
                "R@2": str(retrieval_seen_queries["recall@2"]),
                "R@4": str(retrieval_seen_queries["recall@4"]),
                "R@8": str(retrieval_seen_queries["recall@8"]),
            }
        },
        "unseen_queries": {
            "calibration": {
                "method": mode,
                "value": str(calibration_unseen_queries.cutpoints[mode]["value"]),
                "distance": str(calibration_unseen_queries.cutpoints[mode]["distance"]),
                "precision": str(calibration_unseen_queries.cutpoints[mode]["precision"]),
                "recall": str(calibration_unseen_queries.cutpoints[mode]["recall"]),
                "binary_accuracy": str(calibration_unseen_queries.cutpoints[mode]["binary_accuracy"]),
                "f1": str(calibration_unseen_queries.cutpoints[mode]["f1"])
            },
            "paper": {
                "R@1": str(retrieval_unseen_queries["recall@1"]),
                "R@2": str(retrieval_unseen_queries["recall@2"]),
                "R@4": str(retrieval_unseen_queries["recall@4"]),
                "R@8": str(retrieval_unseen_queries["recall@8"]),
            }
        },
    }

    with open(full_save_path, "w") as o:
        o.write(json.dumps(config_json))


def run(config):
    version = config["version"]
    cprint(f"Version {version}", "yellow")

    for dataset_name, dconf in config["datasets"].items():
        cprint("|- loading dataset", "blue")

        ds_index, ds_seen_queries, ds_unseen_queries = load_data(version, dataset_name)

        ds_index_x, ds_index_y = split_dataset(ds_index)
        ds_seen_queries_x, ds_seen_queries_y = split_dataset(ds_seen_queries)
        ds_unseen_queries_x, ds_unseen_queries_y = split_dataset(ds_unseen_queries)

        for lparams in dconf["losses"]:
            cprint(f"Evaluating {lparams['name']} on {dataset_name}", "green")
            evaluate(
                version=version,
                dataset_name=dataset_name,
                lparams=lparams,
                distance=dconf["distance"],
                ds_index_x=ds_index_x,
                ds_index_y=ds_index_y,
                ds_seen_queries_x=ds_seen_queries_x,
                ds_seen_queries_y=ds_seen_queries_y,
                ds_unseen_queries_x=ds_unseen_queries_x,
                ds_unseen_queries_y=ds_unseen_queries_y,
            )
            cprint(
                f"Successfully finished evaluation of {lparams['name']} on {dataset_name}",
                "yellow",
            )
            print("\n\n")


if __name__ == "__main__":
    # UNCOMMENT IF RUNNING IN VSCODE
    # os.chdir("./similarity/")
    print(os.listdir())  # If config not in path, then change via os.chdir
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", "-c", help="config path")
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
    # should run
