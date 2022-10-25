from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from tensorflow_similarity.losses import (
    XBM,
    CircleLoss,
    MultiSimilarityLoss,
    PNLoss,
    SoftNearestNeighborLoss,
    TripletLoss,
)

LOSSES = {}
LOSSES["circle"] = lambda p: CircleLoss(
    distance=p.get("distance", "cosine"),
    gamma=p.get("gamma", 80.0),
    margin=p.get("margin", 0.40),
)
LOSSES["multisim"] = lambda p: MultiSimilarityLoss(
    distance=p.get("distance", "cosine"),
    alpha=p.get("alpha", 2.0),
    beta=p.get("beta", 40.0),
    epsilon=p.get("epsilon", 0.5),
    lmda=p.get("lmda", 0.5),
    center=p.get("center", 1.0),
)
LOSSES["pn"] = lambda p: PNLoss(
    distance=p.get("distance", "cosine"),
    positive_mining_strategy=p.get("positive_mining", "hard"),
    negative_mining_strategy=p.get("negative_mining", "semi-hard"),
    soft_margin=p.get("soft_margin", True),
    margin=p.get("margin", 0.1),
)
LOSSES["soft_nn"] = lambda p: SoftNearestNeighborLoss(
    distance=p.get("distance", "cosine"),
    margin=p.get("temperature", 1.0),
)
LOSSES["triplet"] = lambda p: TripletLoss(
    distance=p.get("distance", "cosine"),
    positive_mining_strategy=p.get("positive_mining", "hard"),
    negative_mining_strategy=p.get("negative_mining", "semi-hard"),
    soft_margin=p.get("soft_margin", True),
    margin=p.get("margin", 0.1),
)


def make_loss(loss_id: str, params: Mapping[str, Any]) -> tf.keras.Loss:
    try:
        loss = LOSSES[loss_id](params)
    except KeyError as exc:
        raise ValueError(f"Unknown loss name: {loss_id}") from exc

    if params.get("xbm", False):
        return XBM(
            loss=loss,
            memory_size=params.get("memory_size", 1000),
            warmup_steps=params.get("warmup_steps", 100),
        )

    return loss
