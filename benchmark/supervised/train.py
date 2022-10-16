"Supervised loss benchmark"
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections.abc import Mapping
from typing import Any

import nmslib
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

from tensorflow_similarity.schedules import WarmUpCosine
from tensorflow_similarity.utils import tf_cap_memory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def make_model(exp: Experiment) -> tf.keras.Model:
    model = make_architecture(exp.architecture.cid, exp.architecture.params)
    loss = make_loss(exp.loss.cid, exp.loss.params)
    opt = make_optimizer(exp.opt.cid, exp.opt.params, exp.lr_schedule)
    model.compile(optimizer=opt, loss=loss)
    return model


def run(cfg: Mapping[str, Any], filter_pattern: str) -> None:
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
        gc.collect()
        tf.keras.backend.clear_session()
        tf.random.set_seed(random_seed)

        # Make result path
        stub = utils.make_stub(version, exp.run_grp)
        utils.clean_dir(stub)

        # Load the raw dataset
        cprint(f"\n|-loading and preprocessing {exp.dataset.cid}\n", "blue")
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
                f"{exp.dataset.params['name']}",
                f"{exp.architecture.params['name']}-{exp.architecture.params['embedding']}",
                f"{exp.loss.params['name']}",
                f"{exp.opt.params['name']}",
                f"{exp.training.params['name']}",
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

        if "lr_schedule" in exp.training.params:
            batch_size = train_ds.classes_per_batch * train_ds.examples_per_class_per_batch
            total_steps = (train_ds.num_examples // batch_size) * epochs
            wu_steps = int(total_steps * exp.training.params["lr_schedule"]["warmup_pctg"])
            d_steps = total_steps - wu_steps
            exp.lr_schedule = WarmUpCosine(
                initial_learning_rate=exp.opt.params["lr"],
                decay_steps=d_steps,
                warmup_steps=wu_steps,
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

        # TODO(ovallis): Enable updating the nmslib params as part of the SimilarityModel __init__
        model._index.search._serach_index = nmslib.init(method="brute_force", space="cosinesimil")

        try:
            model.reset_index()
        except AttributeError:
            model.create_index()

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


if __name__ == "__main__":
    tf_cap_memory()

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", "-c", help="config path")
    parser.add_argument("--filter", "-f", help="run only the run groups that match the regexp", default=".*")
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config, filter_pattern=args.filter)
