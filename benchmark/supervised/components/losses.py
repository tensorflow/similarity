from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import keras_tuner
import tensorflow as tf

from tensorflow_similarity.losses import (
    XBM,
    CircleLoss,
    MultiSimilarityLoss,
    PNLoss,
    SoftNearestNeighborLoss,
    TripletLoss,
)

from . import utils

LOSSES = {}
LOSSES["circle"] = lambda p, hp: CircleLoss(
    distance=utils.get_param(p, "distance", "cosine", hp),
    gamma=utils.get_param(p, "gamma", 80.0, hp),
    margin=utils.get_param(p, "margin", 0.40, hp),
)
LOSSES["multisim"] = lambda p, hp: MultiSimilarityLoss(
    distance=utils.get_param(p, "distance", "cosine", hp),
    alpha=utils.get_param(p, "alpha", 2.0, hp),
    beta=utils.get_param(p, "beta", 40.0, hp),
    epsilon=utils.get_param(p, "epsilon", 0.5, hp),
    lmda=utils.get_param(p, "lmda", 0.5, hp),
    center=utils.get_param(p, "center", 1.0, hp),
)
LOSSES["pn"] = lambda p, hp: PNLoss(
    distance=utils.get_param(p, "distance", "cosine", hp),
    positive_mining_strategy=utils.get_param(p, "positive_mining", "hard", hp),
    negative_mining_strategy=utils.get_param(p, "negative_mining", "semi-hard", hp),
    margin=utils.get_param(p, "margin", None, hp),
)
LOSSES["soft_nn"] = lambda p, hp: SoftNearestNeighborLoss(
    distance=utils.get_param(p, "distance", "sql2", hp),
    temperature=utils.get_param(p, "temperature", 1.0, hp),
)
LOSSES["triplet"] = lambda p, hp: TripletLoss(
    distance=utils.get_param(p, "distance", "cosine", hp),
    positive_mining_strategy=utils.get_param(p, "positive_mining", "hard", hp),
    negative_mining_strategy=utils.get_param(p, "negative_mining", "semi-hard", hp),
    margin=utils.get_param(p, "margin", None, hp),
)


def make_loss(
    loss_id: str,
    params: Mapping[str, Any],
    train_params: Mapping[str, Any],
    hp: keras_tuner.HyperParameters | None = None,
) -> tf.keras.Loss:
    try:
        loss = LOSSES[loss_id](params, hp)
    except KeyError as exc:
        raise ValueError(f"Unknown loss name: {loss_id}") from exc

    if "xbm" in params:
        xbm_params = params["xbm"]
        if "memory_ratio" in xbm_params:
            mem_size = _mem_size_from_ratio(xbm_params["memory_ratio"], train_params)
        elif "memory_size" in xbm_params:
            mem_size = xbm_params["memory_size"]
        else:
            raise ValueError("One of memory_ratio or memory_size must be set when using xbm")
        return XBM(
            loss=loss,
            memory_size=mem_size,
            warmup_steps=xbm_params.get("warmup_steps", 1000),
        )

    return loss


def _mem_size_from_ratio(mem_ratio, train_params):
    num_examples = train_params["num_examples"]
    batch_size = train_params["classes_per_batch"] * train_params["examples_per_class_per_batch"]
    mem_size = int(num_examples * mem_ratio)
    return max(mem_size, batch_size)
