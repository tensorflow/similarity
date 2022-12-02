from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Mapping
from typing import Any

import keras_tuner
import numpy as np
import tensorflow as tf

from tensorflow_similarity.search import NMSLibSearch

from .architectures import make_architecture
from .experiments import Experiment
from .losses import make_loss
from .optimizers import make_optimizer


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def clean_dir(fpath: str) -> None:
    "delete previous content and recreate dir"
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
    os.makedirs(fpath, exist_ok=True)


def make_stub(version: str, name: str, root_dir: str) -> str:
    return os.path.join(root_dir, version, name)


def make_run_grp(
    dataset_name: str,
    architecture_name: str,
    embedding: int,
    loss_name: str,
    opt_name: str,
) -> str:
    return "_".join([dataset_name, architecture_name, f"emb_{embedding}", loss_name, opt_name])


def set_random_seeds(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def make_model(exp: Experiment, hp: keras_tuner.HyperParameters | None = None) -> tf.keras.Model:
    model = make_architecture(exp.architecture.cid, exp.architecture.params)
    loss = make_loss(exp.loss.cid, exp.loss.params, exp.training.params["train"], hp)
    opt = make_optimizer(exp.opt.cid, exp.opt.params, exp.lr_schedule, hp)
    search = NMSLibSearch(
        distance=loss.distance,
        dim=exp.architecture.params["embedding"],
        method="hnsw",
        index_params={"efConstruction": 100, "M": 15},
    )
    model.compile(optimizer=opt, loss=loss, search=search)
    return model


def get_param(
    params: Mapping[str, Any], key: str, default: Any, hp: keras_tuner.HyperParameters | None = None
) -> str | bool | float:
    if key not in params:
        return default

    value = params[key]
    # If the value is a dict, then it contains hyperparameter tuner parameters
    if isinstance(value, dict):
        if hp is None:
            raise ValueError("must pass a valid keras_tuner.HyperParameters obj to get_param")

        if value["component"] == "int":
            return hp.Int(
                name=key,
                min_value=value["min_value"],
                max_value=value["max_value"],
                step=value.get("step", None),
                sampling=value.get("sampling", "linear"),
                default=value.get("default", None),
            )
        elif value["component"] == "float":
            return hp.Float(
                name=key,
                min_value=value["min_value"],
                max_value=value["max_value"],
                step=value.get("step", None),
                sampling=value.get("sampling", "linear"),
                default=value.get("default", None),
            )
        elif value["component"] == "choice":
            return hp.Choice(
                name=key,
                values=value["values"],
                ordered=value.get("ordered", None),
                default=value.get("default", None),
            )
        elif value["component"] == "boolean":
            return hp.Boolean(
                name=key,
                default=value.get("default", False),
            )
        else:
            raise ValueError(f"Unknown KerasTuner component type: {value['component']}")

    return value
