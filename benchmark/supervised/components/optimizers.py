from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow_addons.optimizers import LAMB

OPTIMIZERS = {}
OPTIMIZERS["adam"] = lambda p, lrs: Adam(
    learning_rate=lrs if lrs else p.get("lr", 0.001),
    beta_1=p.get("beta_1", 0.9),
    beta_2=p.get("beta_2", 0.999),
    epsilon=p.get("epsilon", 1e-07),
    amsgrad=p.get("amsgrad", False),
)
OPTIMIZERS["lamb"] = lambda p, lrs: LAMB(
    learning_rate=lrs if lrs else p.get("lr", 0.001),
    beta_1=p.get("beta_1", 0.9),
    beta_2=p.get("beta_2", 0.999),
    epsilon=p.get("epsilon", 1e-06),
    weight_decay=p.get("weight_decay", 0.0),
    exclude_from_weight_decay=p.get("exclude_from_weight_decay", None),
    exclude_from_layer_adaptation=p.get("exclude_from_layer_adaptation", None),
)
OPTIMIZERS["rmsprop"] = lambda p, lrs: RMSprop(
    learning_rate=lrs if lrs else p.get("lr", 0.001),
    rho=p.get("rho", 0.9),
    momentum=p.get("momentum", 0.0),
    epsilon=p.get("epsilon", 1e-07),
    centered=p.get("centered", False),
)


def make_optimizer(
    opt_id: str,
    params: Mapping[str, Any],
    lr_schedule: LearningRateSchedule | None,
) -> tf.keras.Optimizer:
    try:
        return OPTIMIZERS[opt_id](params, lr_schedule)
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer name: {opt_id}") from exc
