import glob
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from termcolor import cprint
from tqdm.auto import tqdm

from . import utils
from .dataset import Dataset, PreProcessFn
from .types import DatasetConfig, Splits


class TFDSDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, data_dir: str) -> None:
        super().__init__(cfg, data_dir)

    def load_raw_data(self, preproc_fns: PreProcessFn, force: bool = False) -> None:
        if force or not self.xs or not self.ys:
            split = "all"
            ds, ds_info = tfds.load(self.cfg.dataset_id, split=split, with_info=True)

            if self.cfg.x_key not in ds_info.features:
                raise ValueError("x_key not found - available features are:", str(ds_info.features.keys()))
            if self.cfg.y_key not in ds_info.features:
                raise ValueError("y_key not found - available features are:", str(ds_info.features.keys()))

            pb = tqdm(total=ds_info.splits[split].num_examples, desc="converting %s" % split)

            # unpack the feature and labels
            for e in ds:
                self.xs.append(e[self.cfg.x_key])
                self.ys.append(e[self.cfg.y_key])
                pb.update()
            pb.close()

            # Apply preproccessing
            with tf.device("/cpu:0"):
                # TODO(ovallis): batch proccess instead of 1 at a time
                for idx in tqdm(range(len(self.xs)), desc="Preprocessing data"):
                    for p in preproc_fns:
                        self.xs[idx] = p(self.xs[idx])

    def save_serialized_data(self) -> None:
        cprint("|-Saving processed data", "blue")
        self._write_raw_data()
        cprint("|-Saving processed data", "blue")
        utils.save_numpy(self.path, "train.npz", self.splits.train_idxs)
        cprint("|-Saving test idxs", "blue")
        utils.save_numpy(self.path, "test.npz", self.splits.test_idxs)
        for fold_id, fold_data in self.splits.folds.items():
            cprint(f"|-Saving {fold_id} split", "blue")
            utils.save_numpy(self.path, f"{fold_id}.npz", fold_data)

    def _write_raw_data(self) -> None:
        with tf.io.TFRecordWriter(utils._make_fname(self.path, self.tf_record_filename)) as fw:
            for x_, y_ in tqdm(zip(self.xs, self.ys), total=len(self.xs)):
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

    def _decode_processed_data(self, record_bytes: tf.train.Example):
        example = tf.io.parse_single_example(
            record_bytes,
            {
                "x": tf.io.FixedLenFeature([], dtype=tf.string),
                "y": tf.io.FixedLenFeature([], dtype=tf.int64),
            },
        )
        example["x"] = tf.io.parse_tensor(example["x"], tf.float32)
        return example

    def load_serialized_data(self):
        ds = tf.data.TFRecordDataset(os.path.join(self.path, "processed_data.tfrec")).map(self._decode_processed_data)
        self.xs = []
        self.ys = []

        for example in ds.as_numpy_iterator():
            self.xs.append(example["x"])
            self.ys.append(example["y"])

        self.splits = Splits()
        self.splits.train_idxs = list(utils.load_numpy(self.path, "train.npz")["data"])
        self.splits.test_idxs = list(utils.load_numpy(self.path, "test.npz")["data"])
        for fn in glob.glob(os.path.join(self.path, "fold_*.npz")):
            (file_path, filename) = os.path.split(fn)
            fold_name = os.path.splitext(filename)[0]
            self.splits.folds[fold_name] = utils.load_numpy(file_path, filename)["data"].item()
