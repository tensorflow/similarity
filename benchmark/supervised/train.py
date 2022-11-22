"Supervised loss benchmark"
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import tracemalloc
from collections.abc import Mapping
from typing import Any

import tensorflow as tf
import tensorflow.keras.backend
import tensorflow.random
from components import (
    datasets,
    make_architecture,
    make_augmentations,
    make_experiments,
    make_loss,
    make_optimizer,
    metrics,
    utils,
)
from components.experiments import Experiment
from tabulate import tabulate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from termcolor import cprint

from tensorflow_similarity.schedules import WarmupCosineDecay
from tensorflow_similarity.search import NMSLibSearch
from tensorflow_similarity.utils import tf_cap_memory

# from tensorflow_similarity.utils import tf_cap_memory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def make_model(exp: Experiment) -> tf.keras.Model:
    model = make_architecture(exp.architecture.cid, exp.architecture.params)
    loss = make_loss(exp.loss.cid, exp.loss.params)
    opt = make_optimizer(exp.opt.cid, exp.opt.params, exp.lr_schedule)
    search = NMSLibSearch(
        distance=loss.distance,
        dim=exp.architecture.params["embedding"],
        method="hnsw",
        index_params={"efConstruction": 100, "M": 15},
    )
    model.compile(optimizer=opt, loss=loss, search=search)
    return model


def run(config):
    if config.get("tfds_data_dir", None):
        os.environ["TFDS_DATA_DIR"] = config["tfds_data_dir"]

    agg_results = {}
    for dataset_name, dconf in config["datasets"].items():
        if "train_val_splits" not in dconf:
            dconf["train_val_splits"] = {
                "n_splits": 1,
                "val_class_pctg": 0.05,
                "max_val_examples": 10000,
            }

        for architecture_name, aconf in config["architectures"].items():
            for embedding_size in aconf.get("embedding_sizes", [128]):
                aconf["embedding"] = embedding_size
                for loss_name, lconf in config["losses"].items():
                    for opt_name, oconf in config["optimizer"].items():
                        for training_name, tconf in config["training"].items():
                            version = config["version"]
                            pconf = config["preprocess"]
                            aug_conf = config["augmentations"]

                            # Load the raw dataset
                            cprint(f"\n|-loading and preprocessing {dataset_name}\n", "blue")
                            preproc_fns = make_augmentations(pconf)
                            x, y = datasets.load_tf_dataset(dataset_name, dconf, preproc_fns)

                            for fold in range(dconf["train_val_splits"]["n_splits"]):
                                gc.collect()
                                tf.keras.backend.clear_session()

                                tf.random.set_seed(config["random_seed"])

                                headers = [
                                    "dataset_name",
                                    "architecture_name",
                                    "loss_name",
                                    "opt_name",
                                    "training_name",
                                ]
                                row = [
                                    [
                                        f"{dataset_name}",
                                        f"{architecture_name}-{aconf['embedding']}",
                                        f"{loss_name}",
                                        f"{opt_name}",
                                        f"{training_name}",
                                    ]
                                ]
                                print("\n")
                                cprint(tabulate(row, headers=headers), "yellow")

                                ds_splits = datasets.create_splits(x, y, dconf, fold)
                                aug_fns = make_augmentations(aug_conf["train"])
                                cprint("\n|-building train dataset\n", "blue")
                                train_ds = datasets.make_sampler(
                                    ds_splits["train"][0], ds_splits["train"][1], tconf, aug_fns
                                )
                                cprint("\n|-building val dataset\n", "blue")
                                val_ds = datasets.make_sampler(
                                    ds_splits["val"][0], ds_splits["val"][1], tconf, aug_fns
                                )

                                # Build model
                                model = build_model(aconf, lconf, oconf)

                                # Make result path
                                stub = utils.make_stub(
                                    version,
                                    dataset_name,
                                    architecture_name,
                                    aconf["embedding"],
                                    loss_name,
                                    opt_name,
                                    fold,
                                )
                                utils.clean_dir(stub)

                                # Training params
                                callbacks = [
                                    metrics.make_eval_callback(
                                        val_ds,
                                        dconf["eval_callback"]["max_num_queries"],
                                        dconf["eval_callback"]["max_num_targets"],
                                    ),
                                    ModelCheckpoint(
                                        stub,
                                        monitor="val_loss",
                                        save_best_only=True,
                                    ),
                                ]

                                if "steps_per_epoch" in tconf:
                                    steps_per_epoch = tconf["steps_per_epoch"]
                                else:
                                    batch_size = train_ds.classes_per_batch * train_ds.examples_per_class_per_batch
                                    steps_per_epoch = train_ds.num_examples // batch_size

                                if "validation_steps" in tconf:
                                    validation_steps = tconf["validation_steps"]
                                else:
                                    batch_size = val_ds.classes_per_batch * val_ds.examples_per_class_per_batch
                                    validation_steps = val_ds.num_examples // batch_size

                                if "epochs" in tconf:
                                    epochs = tconf["epochs"]
                                else:
                                    epochs = 1000
                                    early_stopping = EarlyStopping(
                                        monitor="val_loss",
                                        patience=5,
                                        verbose=0,
                                        mode="auto",
                                        restore_best_weights=True,
                                    )
                                    callbacks.append(early_stopping)

                                t_msg = [
                                    "\n|-Training",
                                    f"|  - Fold:               {fold}",
                                    f"|  - Num train examples: {train_ds.num_examples}",
                                    f"|  - Num val examples:   {val_ds.num_examples}",
                                    f"|  - Steps per epoch:    {steps_per_epoch}",
                                    f"|  - Epochs:             {epochs}",
                                    f"|  - Validation steps:   {validation_steps}",
                                    "|  - Eval callback",
                                    f"|  -- Num queries:       {len(callbacks[0].queries_known)}",
                                    f"|  -- Num targets:       {len(callbacks[0].targets)}",
                                ]
                                cprint("\n".join(t_msg) + "\n", "green")
                                history = model.fit(
                                    train_ds,
                                    steps_per_epoch=steps_per_epoch,
                                    epochs=epochs,
                                    callbacks=callbacks,
                                    validation_data=val_ds,
                                    validation_steps=validation_steps,
                                )

                                # Evaluation
                                test_aug_fns = make_augmentations(aug_conf["test"])
                                cprint("\n|-building eval dataset\n", "blue")
                                test_x, test_y, class_counts = datasets.make_eval_data(
                                    ds_splits["test"][0], ds_splits["test"][1], test_aug_fns
                                )

                                print("Make Metrics")
                                eval_metrics = metrics.make_eval_metrics(dconf, config["evaluation"], class_counts)

                                try:
                                    model.reset_index()
                                except AttributeError:
                                    model.create_index()
                                print("Add Examples to Index")
                                model.index(test_x, test_y)

                                e_msg = [
                                    "\n|-Evaluate Retriveal Metrics",
                                    f"|  - Fold:              {fold}",
                                    f"|  - Num eval examples: {len(test_x)}",
                                ]
                                cprint("\n".join(e_msg) + "\n", "green")
                                eval_results = model.evaluate_retrieval(
                                    test_x,
                                    test_y,
                                    retrieval_metrics=eval_metrics,
                                )

                                agg_results[os.path.basename(stub)] = eval_results

                                # Save history
                                with open(os.path.join(stub, "history.json"), "w") as o:
                                    o.write(json.dumps(history.history, cls=utils.NpEncoder))

                                # Save eval metrics
                                with open(os.path.join(stub, "eval_metrics.json"), "w") as o:
                                    o.write(json.dumps(eval_results, cls=utils.NpEncoder))

    with open(os.path.join(os.path.dirname(stub), "all_eval_metrics.json"), "w") as o:
        o.write(json.dumps(agg_results, cls=utils.NpEncoder))
=======
def run(cfg: Mapping[str, Any], filter_pattern: str) -> None:
    tracemalloc.start()
    snapshots = []

    if cfg.get("tfds_data_dir", None):
        os.environ["TFDS_DATA_DIR"] = cfg["tfds_data_dir"]

    version = cfg["version"]
    random_seed = cfg["random_seed"]
    preproc_fns = make_augmentations(cfg["preprocess"])
    train_aug_fns = make_augmentations(cfg["augmentations"]["train"])
    test_aug_fns = make_augmentations(cfg["augmentations"]["test"])

    p = re.compile(filter_pattern)
    experiments = [e for e in make_experiments(cfg) if p.match(e.run_grp)]

    for exp in experiments:
        cprint(f"|-{exp.run_grp}", "blue")

    cprint(f"{len(experiments)} Run Groups\n", "blue")
    if input("Would you like to continue: [Y/n] ").lower() != "y":
        cprint("Exit", "red")
        return
    else:
        cprint("Begin Training", "green")

    for exp in experiments:
        tf.random.set_seed(random_seed)

        # Make result path
        stub = utils.make_stub(version, exp.run_grp)
        utils.clean_dir(stub)

        # Load the raw dataset
        cprint(f"\n|-loading and preprocessing {exp.dataset.name}\n", "blue")
        x, y = datasets.load_tf_dataset(exp.dataset, preproc_fns)

        headers = [
            "dataset_name",
            "architecture_name",
            "loss_name",
            "opt_name",
            "training_name",
            "fold",
        ]
        row = [
            [
                f"{exp.dataset.name}",
                f"{exp.architecture.name}-{exp.architecture.params['embedding']}",
                f"{exp.loss.name}",
                f"{exp.opt.name}",
                f"{exp.training.name}",
                f"{exp.fold}",
            ]
        ]
        print("\n")
        cprint(tabulate(row, headers=headers), "yellow")

        ds_splits = datasets.create_splits(x, y, exp.dataset.params, exp.fold)
        cprint("\n|-building train dataset\n", "blue")
        train_ds = datasets.make_sampler(
            ds_splits["train"][0],
            ds_splits["train"][1],
            exp.training.params["train"],
            train_aug_fns,
        )
        cprint("\n|-building val dataset\n", "blue")
        val_ds = datasets.make_sampler(
            ds_splits["val"][0],
            ds_splits["val"][1],
            exp.training.params["val"],
            train_aug_fns,
        )

        # Training params
        callbacks = [
            metrics.make_eval_callback(
                val_ds,
                exp.dataset.params["eval_callback"]["max_num_queries"],
                exp.dataset.params["eval_callback"]["max_num_targets"],
            ),
            ModelCheckpoint(
                stub,
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        if "steps_per_epoch" in exp.training.params:
            steps_per_epoch = exp.training.params["steps_per_epoch"]
        else:
            batch_size = train_ds.classes_per_batch * train_ds.examples_per_class_per_batch
            steps_per_epoch = train_ds.num_examples // batch_size

        if "validation_steps" in exp.training.params:
            validation_steps = exp.training.params["validation_steps"]
        else:
            batch_size = val_ds.classes_per_batch * val_ds.examples_per_class_per_batch
            validation_steps = val_ds.num_examples // batch_size

        if "epochs" in exp.training.params:
            epochs = exp.training.params["epochs"]
        else:
            epochs = 1000
            # TODO(ovallis): expose EarlyStopping params in config
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=5,
                verbose=0,
                mode="auto",
                restore_best_weights=True,
            )
            callbacks.append(early_stopping)

        # TODO(ovallis): break this out into a benchmark component
        if "lr_schedule" in exp.training.params:
            batch_size = train_ds.classes_per_batch * train_ds.examples_per_class_per_batch
            total_steps = (train_ds.num_examples // batch_size) * epochs
            wu_steps = int(total_steps * exp.training.params["lr_schedule"]["warmup_pctg"])
            alpha = exp.training.params["lr_schedule"]["min_lr"] / exp.opt.params["lr"]
            exp.lr_schedule = WarmupCosineDecay(
                max_learning_rate=exp.opt.params["lr"],
                total_steps=total_steps,
                warmup_steps=wu_steps,
                alpha=alpha,
            )

        t_msg = [
            "\n|-Training",
            f"|  - Fold:               {exp.fold}",
            f"|  - Num train examples: {train_ds.num_examples}",
            f"|  - Num val examples:   {val_ds.num_examples}",
            f"|  - Steps per epoch:    {steps_per_epoch}",
            f"|  - Epochs:             {epochs}",
            f"|  - Validation steps:   {validation_steps}",
            "|  - Eval callback",
            f"|  -- Num queries:       {len(callbacks[0].queries_known)}",
            f"|  -- Num targets:       {len(callbacks[0].targets)}",
        ]
        cprint("\n".join(t_msg) + "\n", "green")

        model = make_model(exp)

        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds,
            validation_steps=validation_steps,
        )

        # Evaluation
        cprint("\n|-building eval dataset\n", "blue")
        test_x, test_y, class_counts = datasets.make_eval_data(
            ds_splits["test"][0], ds_splits["test"][1], test_aug_fns
        )

        eval_metrics = metrics.make_eval_metrics(cfg["evaluation"], class_counts)

        del model._index.search
        del model._index.search_type
        model._index.search_type = NMSLibSearch(
            distance=model.loss.distance,
            dim=exp.architecture.params["embedding"],
            method="brute_force",
        )
        model.reset_index()

        model.index(test_x, test_y)

        e_msg = [
            "\n|-Evaluate Retriveal Metrics",
            f"|  - Fold:              {exp.fold}",
            f"|  - Num eval examples: {len(test_x)}",
        ]
        cprint("\n".join(e_msg) + "\n", "green")
        eval_results = model.evaluate_retrieval(
            test_x,
            test_y,
            retrieval_metrics=eval_metrics,
        )

        # Save history
        with open(os.path.join(stub, "history.json"), "w") as o:
            o.write(json.dumps(history.history, cls=utils.NpEncoder))

        # Save eval metrics
        with open(os.path.join(stub, "eval_metrics.json"), "w") as o:
            o.write(json.dumps(eval_results, cls=utils.NpEncoder))

        # Ensure we release all the mem
        for c in callbacks:
            del c
        for e in eval_metrics:
            del e
        del model._index.search
        del model._index.search_type
        del model
        del exp.lr_schedule
        del train_ds._x
        del train_ds._y
        del val_ds._x
        del val_ds._y
        del ds_splits
        del train_ds
        del val_ds
        del test_x
        del test_y
        tf.keras.backend.clear_session()
        gc.collect()

        snapshots.append(tracemalloc.take_snapshot())

        if len(snapshots) == 1:
            top_stats = snapshots[0].statistics("lineno")
        else:
            top_stats = snapshots[-1].compare_to(snapshots[-2], "lineno")

        print("[ Top 10 stats ]")
        for stat in top_stats[:10]:
            print(stat)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", "-c", help="config path")
    parser.add_argument("--filter", "-f", help="run only the run groups that match the regexp", default=".*")
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()

    tf_cap_memory()
    gc.collect()
    tf.keras.backend.clear_session()

    config = json.loads(open(args.config).read())
    run(config, filter_pattern=args.filter)
