from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.types import FloatTensor, IntTensor

from .experiments import Component


# Ensure we only reload the dataset when it's passed new params
@lru_cache(maxsize=1)
def load_tf_dataset(
    cfg: Component,
    preproc_fns: tuple[Callable[[FloatTensor], FloatTensor]],
) -> tuple[FloatTensor, IntTensor]:
    x, y = [], []
    split = "all"
    ds, ds_info = tfds.load(cfg.cid, split=split, with_info=True)

    if cfg.params["x_key"] not in ds_info.features:
        raise ValueError("x_key not found - available features are:", str(ds_info.features.keys()))
    if cfg.params["y_key"] not in ds_info.features:
        raise ValueError("y_key not found - available features are:", str(ds_info.features.keys()))

    pb = tqdm(total=ds_info.splits[split].num_examples, desc="converting %s" % split)

    # unpack the feature and labels
    for e in ds:
        x.append(e[cfg.params["x_key"]])
        y.append(e[cfg.params["y_key"]])
        pb.update()
    pb.close()

    # Apply preproccessing
    with tf.device("/cpu:0"):
        # TODO(ovallis): batch proccess instead of 1 at a time
        for idx in tqdm(range(len(x)), desc="Preprocessing data"):
            for p in preproc_fns:
                x[idx] = p(x[idx])

    return x, y


def create_splits(
    x: Sequence[FloatTensor],
    y: Sequence[IntTensor],
    cfg: Mapping[str, Any],
    fold: int,
) -> dict[str, tuple[FloatTensor, IntTensor]]:
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    # create list of clases to include in the train split
    train_classes = cfg["train_classes"]
    train_classes = list(range(train_classes[0], train_classes[1]))

    # select a slice of the train classes to use for validation
    # ensure we have at least 1 class for validation
    val_len = int(len(train_classes) * cfg["train_val_splits"]["val_class_pctg"])
    val_len = max(1, val_len)
    val_start = val_len * fold
    val_end = val_start + val_len

    # constuct the disjoint sets of val and train classes
    val_classes = set(train_classes[val_start:val_end])
    train_classes = set(train_classes[:val_start] + train_classes[val_end:])

    # create the set of test classes
    test_classes = cfg["test_classes"]
    test_classes = set(range(test_classes[0], test_classes[1]))

    # split tain and test by class ranges
    for example, label in zip(x, y):
        label_val = label.numpy()
        if label_val in train_classes:
            train_x.append(example)
            train_y.append(label)

        if label_val in val_classes:
            val_x.append(example)
            val_y.append(label)

        if label_val in test_classes:
            test_x.append(example)
            test_y.append(label)

    return {"train": (train_x, train_y), "val": (val_x, val_y), "test": (test_x, test_y)}


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
    x: MutableSequence[FloatTensor],
    y: MutableSequence[IntTensor],
    aug_fns: tuple[Any, ...],
) -> tuple[FloatTensor, IntTensor, dict[int, int]]:
    with tf.device("/cpu:0"):
        # TODO(ovallis): batch proccess instead of 1 at a time
        for idx in tqdm(range(len(x)), desc="Preprocessing data"):
            for p in aug_fns:
                x[idx] = p(x[idx])

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {k: v for k, v in zip(unique, counts)}

    return (tf.convert_to_tensor(np.array(x)), tf.convert_to_tensor(np.array(y)), class_counts)
