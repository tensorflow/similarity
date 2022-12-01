from __future__ import annotations

import dataclasses
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from components import utils
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.types import FloatTensor, IntTensor


def load_tf_dataset(
    cfg: DatasetConfig,
    preproc_fns: tuple[Callable[[FloatTensor], FloatTensor]],
) -> tuple[FloatTensor, IntTensor]:
    x, y = [], []
    split = "all"
    ds, ds_info = tfds.load(cfg.dataset_name, split=split, with_info=True)

    if cfg.x_key not in ds_info.features:
        raise ValueError("x_key not found - available features are:", str(ds_info.features.keys()))
    if cfg.y_key not in ds_info.features:
        raise ValueError("y_key not found - available features are:", str(ds_info.features.keys()))

    pb = tqdm(total=ds_info.splits[split].num_examples, desc="converting %s" % split)

    # unpack the feature and labels
    for e in ds:
        x.append(e[cfg.x_key])
        y.append(e[cfg.y_key])
        pb.update()
    pb.close()

    # Apply preproccessing
    with tf.device("/cpu:0"):
        # TODO(ovallis): batch proccess instead of 1 at a time
        for idx in tqdm(range(len(x)), desc="Preprocessing data"):
            for p in preproc_fns:
                x[idx] = p(x[idx])

    return x, y


def _create_folds(train_classes: list[int], cfg: DatasetConfig) -> dict[str, defaultdict[str, list[Any]]]:
    folds = {}
    # ensure we have at least 1 class for validation
    num_val_folds = int(1.0 / cfg.train_val_splits.val_class_pctg)
    num_val_folds = max(1, num_val_folds)
    num_splits = cfg.train_val_splits.num_splits
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
        folds[f"fold_{i}"] = defaultdict(list)
        folds[f"fold_{i}"]["train_classes"] = t
        folds[f"fold_{i}"]["val_classes"] = v

    return folds


def create_splits(
    x: Sequence[FloatTensor],
    y: Sequence[IntTensor],
    cfg: DatasetConfig,
) -> dict[str, tuple[FloatTensor, IntTensor]]:
    train_idxs, test_idxs = [], []

    # create list of clases to include in the train split
    train_classes = cfg.train_classes
    train_classes = list(range(train_classes[0], train_classes[1]))
    random.shuffle(train_classes)

    # create the set of test classes
    test_classes = cfg.test_classes
    test_classes = set(range(test_classes[0], test_classes[1]))

    folds = _create_folds(train_classes, cfg)

    # split train, test, and folds by class ranges
    for idx, label in enumerate(y):
        lbl = label.numpy()
        if lbl in train_classes:
            train_idxs.append(idx)

        if lbl in test_classes:
            test_idxs.append(idx)

        for fold_id, fold_classes in folds.items():
            if lbl in fold_classes["train_classes"]:
                folds[fold_id]["train_idxs"].append(idx)

            if lbl in fold_classes["val_classes"]:
                folds[fold_id]["val_idxs"].append(idx)

    processed_data = {"x": x, "y": y}
    return {
        "processed_data": processed_data,
        "train_idxs": train_idxs,
        "test_idxs": test_idxs,
        "folds": folds,
    }


def save_processed_data(path: str, filename: str, x: FloatTensor, y: IntTensor) -> None:
    with tf.io.TFRecordWriter(utils._make_fname(path, filename)) as fw:
        for x_, y_ in tqdm(zip(x, y), total=len(x)):
            x_ = tf.io.serialize_tensor(x_)
            record_bytes = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_.numpy()])),
                        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y_.numpy()])),
                    }
                )
            ).SerializeToString()
            fw.write(record_bytes)


def decode_processed_data(record_bytes):
    example = tf.io.parse_single_example(
        record_bytes,
        {
            "x": tf.io.FixedLenFeature([], dtype=tf.string),
            "y": tf.io.FixedLenFeature([], dtype=tf.int64),
        },
    )
    example["x"] = tf.io.parse_tensor(example["x"], tf.float32)
    return example


# TODO(ovallis): aug_fns type should be tuple[Callable[[FloatTensor], FloatTensor]], but
# mypy doesn't recogonize the return types of the callabels.
def make_sampler(
    x: Sequence[FloatTensor],
    y: Sequence[IntTensor],
    cfg: dict[str, Any],
    aug_fns: tuple[Any, ...],
) -> MultiShotMemorySampler:
    def augmentation_fn(x, y, *args):
        for a in aug_fns:
            x = a(x)
        return x, y

    return MultiShotMemorySampler(
        x,
        y,
        classes_per_batch=cfg.get("classes_per_batch", 2),
        examples_per_class_per_batch=cfg.get("examples_per_class_per_batch", 2),
        augmenter=augmentation_fn,
    )


# TODO(ovallis): aug_fns type should be tuple[Callable[[FloatTensor], FloatTensor]], but
# mypy doesn't recogonize the return types of the callabels.
def make_eval_data(
    x: Sequence[FloatTensor],
    y: Sequence[IntTensor],
    aug_fns: tuple[Any, ...],
) -> tuple[FloatTensor, IntTensor, dict[int, int]]:
    aug_x = []
    with tf.device("/cpu:0"):
        # TODO(ovallis): batch proccess instead of 1 at a time
        for idx in tqdm(range(len(x)), desc="Preprocessing data"):
            for p in aug_fns:
                aug_x.append(p(x[idx]))

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {k: v for k, v in zip(unique, counts)}

    return (tf.convert_to_tensor(np.array(aug_x)), tf.convert_to_tensor(np.array(y)), class_counts)


@dataclasses.dataclass
class TrainValSplit:
    num_splits: int
    val_class_pctg: float


@dataclasses.dataclass
class EvalCallback:
    max_num_queries: int
    max_num_targets: int


@dataclasses.dataclass
class DatasetConfig:
    config_id: str
    dataset_name: str
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


def make_dataset_configs(cfg: Mapping[str, Any]) -> list[DatasetConfig]:
    dataset_cfgs = []

    for dn, dcfg in cfg.items():
        if "train_val_splits" in dcfg:
            tv_split = TrainValSplit(**dcfg["train_val_splits"])
            dcfg.pop("train_val_splits")
        else:
            tv_split = None

        if "eval_callback" in dcfg:
            ev_cb = EvalCallback(**dcfg["eval_callback"])
            dcfg.pop("eval_callback")
        else:
            ev_cb = None

        dataset_cfgs.append(
            DatasetConfig(
                config_id=dn,
                train_val_splits=tv_split,
                eval_callback=ev_cb,
                **dcfg,
            )
        )

    return dataset_cfgs
