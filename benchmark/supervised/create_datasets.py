"Supervised loss benchmark"
from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from typing import Any

import tensorflow as tf
import tensorflow.random
from components import datasets, make_augmentations, utils
from termcolor import cprint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def run(cfg: Mapping[str, Any], filter_pattern: str) -> None:

    if cfg.get("tfds_data_dir", None):
        os.environ["TFDS_DATA_DIR"] = cfg["tfds_data_dir"]

    version = cfg["version"]
    random_seed = cfg["random_seed"]
    output_dir = cfg["dataset_dir"]
    preproc_fns = make_augmentations(cfg["preprocess"])

    p = re.compile(filter_pattern)
    data_dir = os.path.join(output_dir, version)
    dataset_cfgs = []
    for name, cfg in cfg["datasets"].items():
        if p.match(name):
            dataset_cfgs.append(datasets.utils.make_dataset_config(name, cfg))

    for d in dataset_cfgs:
        cprint(f"|-{d.name}", "blue")

    cprint(f"{len(dataset_cfgs)} Datasets\n", "blue")
    if input("Would you like to continue: [Y/n] ").lower() != "y":
        cprint("Exit", "red")
        return
    else:
        cprint("Building datasets", "green")

    for dcfg in dataset_cfgs:
        d = datasets.utils.make_dataset(dcfg, data_dir)
        utils.set_random_seeds(random_seed)

        # Make result path
        cprint(f"\n|-Clearing all files in {d.path}", "blue")
        utils.clean_dir(d.path)

        # Load the raw dataset
        cprint(f"\n|-Loading and preprocessing {d.name}\n", "blue")
        d.load_raw_data(preproc_fns)
        d.split_raw_data()
        d.save_serialized_data()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument("--config", "-c", help="config path")
    parser.add_argument("--filter", "-f", help="run only the config ids that match the regexp", default=".*")
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()

    config = json.loads(open(args.config).read())
    run(config, filter_pattern=args.filter)
