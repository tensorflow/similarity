"Supervised loss benchmark"
import argparse
import gc
import json
import os

import tensorflow as tf
import tensorflow.keras.backend
import tensorflow.random
from components import (
    datasets,
    make_architecture,
    make_augmentations,
    make_loss,
    make_optimizer,
    metrics,
    utils,
)
from tabulate import tabulate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from termcolor import cprint

from tensorflow_similarity.utils import tf_cap_memory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def build_model(aconf, lconf, oconf):
    model = make_architecture(aconf)
    loss = make_loss(lconf)
    opt = make_optimizer(oconf)
    model.compile(optimizer=opt, loss=loss)
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

                            headers = ["dataset_name", "architecture_name", "loss_name", "opt_name", "training_name"]
                            row = [
                                [
                                    f"{dataset_name}",
                                    f"{architecture_name}",
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
                            val_ds = datasets.make_sampler(ds_splits["val"][0], ds_splits["val"][1], tconf, aug_fns)

                            # Build model
                            model = build_model(aconf, lconf, oconf)

                            # Make result path
                            stub = utils.make_stub(
                                version,
                                dataset_name,
                                architecture_name,
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
                                    monitor="val_loss", patience=5, verbose=0, mode="auto", restore_best_weights=True
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
                            cprint('\n'.join(t_msg)+"\n", "green")
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
                            cprint('\n'.join(e_msg)+"\n", "green")
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


if __name__ == "__main__":
    tf_cap_memory()

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", "-c", help="config path")
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
