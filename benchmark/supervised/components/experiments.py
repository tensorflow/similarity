from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from itertools import product
from typing import Any

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from . import utils


@dataclasses.dataclass(eq=True, frozen=True)
class Component:
    cid: str
    params: Mapping[str, Any]

    def __hash__(self):
        return hash((self.cid, self.params["name"]))


@dataclasses.dataclass
class Experiment:
    run_grp: str
    fold: int
    dataset: Component
    architecture: Component
    loss: Component
    opt: Component
    training: Component
    lr_schedule: LearningRateSchedule | None = None


def make_experiments(cfg: Mapping[str, Any]) -> list[Experiment]:
    experiments = []

    # Generate the cross product of all the experiment params.
    for (did, dcfg), (aid, acfg), (lid, lcfg), (oid, ocfg), t in product(
        cfg["datasets"].items(),
        cfg["architectures"].items(),
        cfg["losses"].items(),
        cfg["optimizer"].items(),
        cfg["training"],
    ):
        if "train_val_splits" not in dcfg:
            dcfg["train_val_splits"] = {
                "n_splits": 1,
                "val_class_pctg": 0.05,
                "max_val_examples": 10000,
            }
        dataset = Component(cid=did, params=dcfg)
        loss = Component(cid=lid, params=lcfg)
        opt = Component(cid=oid, params=ocfg)
        training = Component(cid=t["name"], params=t)

        for embedding_size in acfg.get("embedding_sizes", [128]):
            acfg["embedding"] = embedding_size
            architecture = Component(cid=aid, params=acfg)

            for fold in range(dataset.params["train_val_splits"]["n_splits"]):
                run_grp = utils.make_run_grp(
                    dataset.params["name"],
                    architecture.params["name"],
                    architecture.params["embedding"],
                    loss.params["name"],
                    opt.params["name"],
                    fold,
                )
                experiments.append(
                    Experiment(
                        run_grp=run_grp,
                        fold=fold,
                        dataset=dataset,
                        architecture=architecture,
                        loss=loss,
                        opt=opt,
                        training=training,
                    )
                )
    return experiments
