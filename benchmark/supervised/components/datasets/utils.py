from __future__ import annotations

import copy
import io
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from tensorflow_similarity.samplers import MultiShotMemorySampler
from tensorflow_similarity.types import FloatTensor, IntTensor

from .dataset import Dataset
from .tfds import TFDSDataset
from .types import DatasetConfig, EvalCallback, TrainValSplit

DATASETS = {}
DATASETS["tfds"] = TFDSDataset


def save_numpy(path: str, filename: str, data: Mapping[str, Any], compression: bool = True) -> None:
    """Serializes to disk.

    Args:
        path: where to store the data.
        data: data to be serialized and stored.
        compression: Compress data. Defaults to True.
    """
    # Writing to a buffer to avoid read error in np.savez when using GFile.
    # See: https://github.com/tensorflow/tensorflow/issues/32090
    io_buffer = io.BytesIO()
    if compression:
        np.savez_compressed(
            io_buffer,
            data=data,
        )
    else:
        np.savez(
            io_buffer,
            data=data,
        )

    with tf.io.gfile.GFile(_make_fname(path, filename), "wb+") as f:
        f.write(io_buffer.getvalue())


def load_numpy(path: str, filename: str) -> int:
    """load data

    Args:
        path: which directory to use to store the data.
        filename: filename.

    Returns:
        data
    """
    fname = _make_fname(path, filename, check_file_exits=True)
    with tf.io.gfile.GFile(fname, "rb") as gfp:
        data = np.load(gfp, allow_pickle=True)
    return data


def _make_fname(path: str, filename: str, check_file_exits: bool = False) -> str:
    if not tf.io.gfile.exists(path):
        raise ValueError("Path doesn't exist")
    fname = os.path.join(path, filename)

    # only for loading
    if check_file_exits and not tf.io.gfile.exists(fname):
        raise ValueError("File not found")
    return fname


def make_dataset(dataset_cfg: DatasetConfig, data_dir: str) -> Dataset:
    ds = DATASETS[dataset_cfg.component](dataset_cfg, data_dir)
    return ds


def make_dataset_config(name: str, params: Mapping[str, Any]) -> DatasetConfig:
    dcfg = copy.deepcopy(params)
    dcfg["name"] = name

    if "train_val_splits" in dcfg and isinstance(dcfg["train_val_splits"], Mapping):
        dcfg["train_val_splits"] = TrainValSplit(**dcfg["train_val_splits"])
    else:
        dcfg["train_val_splits"] = None
    if "eval_callback" in dcfg and isinstance(dcfg["eval_callback"], Mapping):
        dcfg["eval_callback"] = EvalCallback(**dcfg["eval_callback"])
    else:
        dcfg["eval_callback"] = None

    dcfg_obj = DatasetConfig.from_dict(dcfg)

    return dcfg_obj


@lru_cache(maxsize=1)
def load_serialized_dataset(dataset_cfg: DatasetConfig, data_dir: str) -> Dataset:
    ds = DATASETS[dataset_cfg.component](dataset_cfg, data_dir)
    ds.load_serialized_data()

    return ds


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
