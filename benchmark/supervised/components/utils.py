from __future__ import annotations

import json
import os
import shutil

import numpy as np

BENCHMARK_DIR = "benchmark_results"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def clean_dir(fpath: str) -> None:
    "delete previous content and recreate dir"
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
    os.makedirs(fpath, exist_ok=True)


def make_stub(version: str, run_grp: str) -> str:
    return os.path.join(BENCHMARK_DIR, version, run_grp)


def make_run_grp(
    dataset_name: str,
    architecture_name: str,
    embedding: int,
    loss_name: str,
    opt_name: str,
    fold: int,
) -> str:
    return "_".join([dataset_name, architecture_name, f"emb_{embedding}", loss_name, opt_name, f"fold_{fold}"])
