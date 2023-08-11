from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from typing import Any


@dataclasses.dataclass
class Fold:
    training_classes: list[int] = dataclasses.field(default_factory=list)
    val_classes: list[int] = dataclasses.field(default_factory=list)
    train_idxs: list[int] = dataclasses.field(default_factory=list)
    val_idxs: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Splits:
    train_idxs: list[int] = dataclasses.field(default_factory=list)
    test_idxs: list[int] = dataclasses.field(default_factory=list)
    folds: Mapping[str, Fold] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainValSplit:
    num_splits: int
    val_class_pctg: float


@dataclasses.dataclass
class EvalCallback:
    max_num_queries: int
    max_num_targets: int


@dataclasses.dataclass(eq=True, frozen=True)
class DatasetConfig:
    name: str
    component: str
    dataset_id: str
    x_key: str
    y_key: str
    train_classes: Sequence[int]
    test_classes: Sequence[int]
    train_val_splits: TrainValSplit | None = None
    eval_callback: EvalCallback | None = None

    def __post_init__(self):
        if self.train_val_splits is None:
            self.train_val_splits = TrainValSplit(
                num_splits=0,
                val_class_pctg=0.0,
            )

    def __hash__(self):
        return hash((self.name, self.component, self.dataset_id, self.x_key, self.y_key))

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> DatasetConfig:
        """Create a DatasetConfig from a dictionary.

        This supports kwargs that are not part of the DatasetConfig class.
        """
        kwargs = {k: v for k, v in params.items() if k in inspect.signature(cls).parameters}
        return cls(**kwargs)
