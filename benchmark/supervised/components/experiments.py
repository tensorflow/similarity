from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from itertools import product
from typing import Any

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from . import datasets, utils


@dataclasses.dataclass(eq=True, frozen=True)
class Component:
    cid: str
    name: str
    params: Mapping[str, Any]

    def __hash__(self):
        return hash((self.cid, self.name))


@dataclasses.dataclass
class Experiment:
    run_grp: str
    path: str
    dataset: Component
    architecture: Component
    loss: Component
    opt: Component
    training: Component
    lr_schedule: LearningRateSchedule | None = None


def make_experiments(cfg: Mapping[str, Any], output_dir: str) -> list[Experiment]:
    experiments = []

    # Generate the cross product of all the experiment params.
    for (dn, dcfg), (an, acfg), (ln, lcfg), (on, ocfg), tcfg in product(
        cfg["datasets"].items(),
        cfg["architectures"].items(),
        cfg["losses"].items(),
        cfg["optimizer"].items(),
        cfg["training"],
    ):
        dataset = datasets.utils.make_dataset_config(name=dn, params=dcfg)
        loss = Component(cid=lcfg["component"], name=ln, params=lcfg)
        opt = Component(cid=ocfg["component"], name=on, params=ocfg)
        training = Component(cid="", name=tcfg["name"], params=tcfg)

        for embedding_size in acfg.get("embedding_sizes", [128]):
            acfg["embedding"] = embedding_size
            architecture = Component(cid=acfg["component"], name=an, params=acfg)

            run_grp = utils.make_run_grp(
                dataset.name,
                architecture.name,
                architecture.params["embedding"],
                loss.name,
                opt.name,
            )
            experiments.append(
                Experiment(
                    run_grp=run_grp,
                    path=os.path.join(output_dir, run_grp),
                    dataset=dataset,
                    architecture=architecture,
                    loss=loss,
                    opt=opt,
                    training=training,
                )
            )
    return experiments
