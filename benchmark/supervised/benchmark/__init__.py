"Shared functions"
import numpy as np
from pathlib import Path
import shutil


def load_dataset(version, dataset_name, shard):
    path = "datasets/%s/%s/%s.npz" % (version, dataset_name, shard)
    d = np.load(path)
    return d['x'], d['y']


def clean_dir(fpath):
    "delete previous content and recreate dir"
    dpath = Path(fpath)
    if dpath.exists():
        shutil.rmtree(fpath)
    dpath = dpath.mkdir(parents=True)
