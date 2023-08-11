from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod

# TODO(ovallis): python 3.8 doesn't support pep585 for type Aliases
# from collections.abc import Callable, Sequence
from typing import Callable, Sequence

from sklearn.model_selection import KFold

from tensorflow_similarity.types import FloatTensor, IntTensor

from .types import DatasetConfig, Fold, Splits

PreProcessFn = Sequence[Callable[[FloatTensor], FloatTensor]]


class Dataset(ABC):
    def __init__(self, cfg: DatasetConfig, data_dir: str) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.path = os.path.join(data_dir, cfg.name)

        self.xs = []
        self.ys = []
        self.splits = Splits()
        self.tf_record_filename = "processed_data.tfrec"
        self.train_idxs_filename = "train.npz"
        self.test_idxs_filename = "test.npz"

    @abstractmethod
    def load_raw_data(self, preproc_fns: PreProcessFn) -> None:
        pass

    @property
    def fold_ids(self):
        return self.splits.folds.keys()

    def get_train_ds(self) -> tuple(FloatTensor | IntTensor):
        return (
            [self.xs[idx] for idx in self.splits.train_idxs],
            [self.ys[idx] for idx in self.splits.train_idxs],
        )

    def get_fold_ds(self, fold_id: str) -> dict[str, tuple(FloatTensor | IntTensor)]:
        if fold_id not in self.splits.folds.keys():
            raise ValueError(f"Fold '{fold_id}' does not exist.")

        train_idxs = self.splits.folds[fold_id].train_idxs
        val_idxs = self.splits.folds[fold_id].val_idxs

        return {
            "train": ([self.xs[idx] for idx in train_idxs], [self.ys[idx] for idx in train_idxs]),
            "val": ([self.xs[idx] for idx in val_idxs], [self.ys[idx] for idx in val_idxs]),
        }

    def get_test_ds(self) -> tuple(FloatTensor | IntTensor):
        return (
            [self.xs[idx] for idx in self.splits.test_idxs],
            [self.ys[idx] for idx in self.splits.test_idxs],
        )

    def _create_folds(self, train_classes: list[int]) -> dict[str, Fold]:
        folds = {}
        # ensure we have at least 1 class for validation
        num_val_folds = int(1.0 / self.cfg.train_val_splits.val_class_pctg)
        num_val_folds = max(1, num_val_folds)
        num_splits = self.cfg.train_val_splits.num_splits
        kfold = KFold(n_splits=num_val_folds, shuffle=True)

        if num_val_folds < num_splits:
            raise ValueError(
                (
                    f"The number of train val splits '{num_splits}' is greater than "
                    f"the number of disjoint folds '{num_val_folds}'. Try reducing the val_class_pctg."
                )
            )

        for i, (t, v) in enumerate(kfold.split(train_classes)):
            if i >= num_splits:
                break
            folds[f"fold_{i}"] = Fold()
            folds[f"fold_{i}"].train_classes = t
            folds[f"fold_{i}"].val_classes = v

        return folds

    def split_raw_data(self) -> None:
        if self.xs is None or self.ys is None:
            print("x or y are empty. You must first run load_raw_data().")
            return

        # create list of clases to include in the train split
        train_classes = self.cfg.train_classes
        train_classes = list(range(train_classes[0], train_classes[1]))
        random.shuffle(train_classes)

        # create the set of test classes
        test_classes = self.cfg.test_classes
        test_classes = set(range(test_classes[0], test_classes[1]))

        self.splits.folds = self._create_folds(train_classes)

        # split train, test, and folds by class ranges
        for idx, label in enumerate(self.ys):
            lbl = label.numpy()
            if lbl in train_classes:
                self.splits.train_idxs.append(idx)

            if lbl in test_classes:
                self.splits.test_idxs.append(idx)

            for fold_id, fold in self.splits.folds.items():
                if lbl in fold.train_classes:
                    self.splits.folds[fold_id].train_idxs.append(idx)

                if lbl in fold.val_classes:
                    self.splits.folds[fold_id].val_idxs.append(idx)

    @abstractmethod
    def save_serialized_data(self) -> None:
        pass

    @abstractmethod
    def load_serialized_data(self) -> None:
        pass
