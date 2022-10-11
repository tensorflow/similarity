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


def clean_dir(fpath):
    "delete previous content and recreate dir"
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
    os.makedirs(fpath, exist_ok=True)


def make_stub(version, dataset_name, architecture_name, loss_name, opt_name, fold_id):
    run_grp = "_".join([dataset_name, architecture_name, loss_name, opt_name, f"{fold_id}"])
    return os.path.join(BENCHMARK_DIR, version, run_grp)
