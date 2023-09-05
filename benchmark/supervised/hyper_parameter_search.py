"Supervised loss benchmark"
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections.abc import Mapping
from functools import partial
from typing import Any

import keras_tuner
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend
import tensorflow.random
from components import datasets, make_augmentations, make_experiments, metrics, utils
from tabulate import tabulate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from termcolor import cprint

from tensorflow_similarity.schedules import WarmupCosineDecay
from tensorflow_similarity.utils import tf_cap_memory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def run(cfg: Mapping[str, Any], filter_pattern: str) -> None:
    if cfg.get("tfds_data_dir", None):
        os.environ["TFDS_DATA_DIR"] = cfg["tfds_data_dir"]

    version = cfg["version"]
    random_seed = cfg["random_seed"]
    train_aug_fns = make_augmentations(cfg["augmentations"]["train"])
    test_aug_fns = make_augmentations(cfg["augmentations"]["test"])

    data_dir = os.path.join(cfg["dataset_dir"], version)
    hp_dir = os.path.join(cfg["hyperparameter_dir"], version)

    p = re.compile(filter_pattern)
    experiments = [e for e in make_experiments(cfg, hp_dir) if p.match(e.run_grp)]

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

        # Load the raw dataset
        cprint(f"\n|-loading preprocessed {exp.dataset.name}\n", "blue")
        ds = datasets.utils.load_serialized_dataset(exp.dataset, data_dir)

        headers = [
            "dataset_name",
            "architecture_name",
            "loss_name",
            "opt_name",
            "training_name",
        ]
        row = [
            [
                f"{exp.dataset.name}",
                f"{exp.architecture.name}-{exp.architecture.params['embedding']}",
                f"{exp.loss.name}",
                f"{exp.opt.name}",
                f"{exp.training.name}",
            ]
        ]
        cprint(tabulate(row, headers=headers), "yellow")

        for fid in ds.fold_ids:
            cprint(f"|-fold {fid}", "blue")
            fold_path = os.path.join(exp.path, fid)

            # Make result path
            utils.clean_dir(fold_path)

            fold_ds = ds.get_fold_ds(fid)

            cprint("\n|-building train dataset\n", "blue")
            train_x = tf.constant(np.array(fold_ds["train"][0]))
            train_y = tf.constant(np.array(fold_ds["train"][1]))
            train_ds = datasets.utils.make_sampler(
                train_x,
                train_y,
                exp.training.params["train"],
                train_aug_fns,
            )
            exp.training.params["train"]["num_examples"] = train_x.shape[0]

            cprint("\n|-building val dataset\n", "blue")
            val_x = tf.constant(np.array(fold_ds["val"][0]))
            val_y = tf.constant(np.array(fold_ds["val"][1]))
            val_ds = datasets.utils.make_sampler(
                val_x,
                val_y,
                exp.training.params["val"],
                train_aug_fns,
            )
            exp.training.params["val"]["num_examples"] = val_x.shape[0]

            # Training params
            callbacks = [
                metrics.make_eval_callback(
                    val_x,
                    val_y,
                    train_aug_fns,
                    exp.dataset.eval_callback.max_num_queries,
                    exp.dataset.eval_callback.max_num_targets,
                ),
                ModelCheckpoint(
                    fold_path,
                    monitor="map@R",
                    mode="max",
                    save_best_only=True,
                ),
                EarlyStopping(
                    monitor="map@R",
                    patience=5,
                    verbose=0,
                    mode="max",
                    restore_best_weights=True,
                ),
            ]

            if "steps_per_epoch" in exp.training.params:
                steps_per_epoch = exp.training.params["steps_per_epoch"]
            else:
                train_params = exp.training.params["train"]
                batch_size = train_params.get("classes_per_batch", 2) * train_params.get(
                    "examples_per_class_per_batch", 2
                )
                steps_per_epoch = train_params["num_examples"] // batch_size

            if "validation_steps" in exp.training.params:
                validation_steps = exp.training.params["validation_steps"]
            else:
                val_params = exp.training.params["val"]
                batch_size = val_params.get("classes_per_batch", 2) * val_params.get("examples_per_class_per_batch", 2)
                validation_steps = val_params["num_examples"] // batch_size

            if "epochs" in exp.training.params:
                epochs = exp.training.params["epochs"]
            else:
                epochs = 1000

            # TODO(ovallis): break this out into a benchmark component
            if "lr_schedule" in exp.training.params:
                total_steps = steps_per_epoch * epochs
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
                f"|  - Fold:               {fid}",
                f"|  - Num train examples: {exp.training.params['train']['num_examples']}",
                f"|  - Num val examples:   {exp.training.params['val']['num_examples']}",
                f"|  - Steps per epoch:    {steps_per_epoch}",
                f"|  - Epochs:             {epochs}",
                f"|  - Validation steps:   {validation_steps}",
                "|  - Eval callback",
                f"|  -- Num queries:       {len(callbacks[0].queries_known)}",
                f"|  -- Num targets:       {len(callbacks[0].targets)}",
            ]
            cprint("\n".join(t_msg) + "\n", "green")

            tuner = keras_tuner.Hyperband(
                hypermodel=partial(utils.make_model, exp),
                objective=keras_tuner.Objective("map@R", "max"),
                max_epochs=10,
                factor=3,
                overwrite=True,
                seed=random_seed,
                directory=fold_path,
                project_name="hyperband",
            )

            cprint(tuner.search_space_summary(), "blue")

            tuner.search(
                train_ds,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=val_ds,
                validation_steps=validation_steps,
            )

            models = tuner.get_best_models(num_models=2)
            if not models:
                print("No models found")
                continue

            model = models[0]
            cprint(model.summary(), "blue")
            cprint(tuner.results_summary(), "green")

            # Evaluation
            cprint("\n|-building eval dataset\n", "blue")
            test_x, test_y = ds.get_test_ds()
            test_x, test_y, class_counts = datasets.utils.make_eval_data(test_x, test_y, test_aug_fns)

            eval_metrics = metrics.make_eval_metrics(cfg["evaluation"], class_counts)

            model.reset_index()

            model.index(test_x, test_y)

            e_msg = [
                "\n|-Evaluate Retriveal Metrics",
                f"|  - Fold:              {fid}",
                f"|  - Num eval examples: {len(test_x)}",
            ]
            cprint("\n".join(e_msg) + "\n", "green")
            eval_results = model.evaluate_retrieval(
                test_x,
                test_y,
                retrieval_metrics=eval_metrics,
            )

            eval_results["keras_tuner_best_params"] = tuner.get_best_hyperparameters()[0].values

            # Save eval metrics
            with open(os.path.join(fold_path, "eval_metrics.json"), "w") as o:
                o.write(json.dumps(eval_results, cls=utils.NpEncoder))

            # Save model
            best_model_path = os.path.join(fold_path, "best_model")
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            model.save(best_model_path, save_index=False)

            # Ensure we release all the mem
            tf.keras.backend.clear_session()
            gc.collect()

        del ds
        gc.collect()


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
