from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import keras_tuner
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow_addons.optimizers import LAMB

from . import utils

OPTIMIZERS = {}
OPTIMIZERS["adam"] = lambda p, lrs, hp: Adam(
    learning_rate=lrs if lrs else utils.get_param(p, "lr", 0.001, hp),
    beta_1=utils.get_param(p, "beta_1", 0.9, hp),
    beta_2=utils.get_param(p, "beta_2", 0.999, hp),
    epsilon=utils.get_param(p, "epsilon", 1e-07, hp),
    amsgrad=utils.get_param(p, "amsgrad", False, hp),
)
OPTIMIZERS["lamb"] = lambda p, lrs, hp: LAMB(
    learning_rate=lrs if lrs else utils.get_param(p, "lr", 0.001, hp),
    beta_1=utils.get_param(p, "beta_1", 0.9, hp),
    beta_2=utils.get_param(p, "beta_2", 0.999, hp),
    epsilon=utils.get_param(p, "epsilon", 1e-06, hp),
    weight_decay=utils.get_param(p, "weight_decay", 0.0, hp),
    exclude_from_weight_decay=utils.get_param(p, "exclude_from_weight_decay", None, hp),
    exclude_from_layer_adaptation=utils.get_param(p, "exclude_from_layer_adaptation", None, hp),
)
OPTIMIZERS["rmsprop"] = lambda p, lrs, hp: RMSprop(
    learning_rate=lrs if lrs else utils.get_param(p, "lr", 0.001, hp),
    rho=utils.get_param(p, "rho", 0.9, hp),
    momentum=utils.get_param(p, "momentum", 0.0, hp),
    epsilon=utils.get_param(p, "epsilon", 1e-07, hp),
    centered=utils.get_param(p, "centered", False, hp),
)


def make_optimizer(
    opt_id: str,
    params: Mapping[str, Any],
    lr_schedule: LearningRateSchedule | None,
    hp: keras_tuner.HyperParameters | None = None,
) -> tf.keras.Optimizer:
    try:
        return OPTIMIZERS[opt_id](params, lr_schedule, hp)
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer name: {opt_id}") from exc
