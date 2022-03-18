"Build datasets benchmarks"
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from termcolor import cprint
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List

print(tfds.__version__)

from benchmark import clean_dir

tf.config.set_visible_devices([], 'GPU')
cprint('Tensorflow set to CPU', 'green')


@dataclass
class DatasetConfig:
    x_key: str
    y_key: str
    splits: List[str]
    train_classes: List[int]
    test_classes: List[int]
    img_size: int
    index_shots: int
    query_shots: int


class DatasetBuilder:
    def __init__(self, dconf, dataset_name, version):
        self.conf = DatasetConfig(
            x_key=dconf['x_key'],
            y_key=dconf['y_key'],
            splits=dconf['splits'],
            train_classes=dconf['train_classes'],
            test_classes=dconf['test_classes'],
            img_size=dconf['shape'][0],
            index_shots=dconf['index_shots'],
            query_shots=dconf['query_shots'],
        )

        self.dataset_name = dataset_name
        self.version = version

    def validate_info(self, ds_info, x_key, y_key):
        if x_key not in ds_info.features:
            raise ValueError("x_key not found - available features are:",
                             str(ds_info.features.keys()))
        if y_key not in ds_info.features:
            raise ValueError("y_key not found - available features are:",
                             str(ds_info.features.keys()))

    def merge_dataset_splits(self):
        x = []
        y = []
        for split in self.conf.splits:
            ds, ds_info = tfds.load(self.dataset_name,
                                    split=split,
                                    with_info=True)

            self.validate_info(ds_info, self.conf.x_key, self.conf.y_key)

            pb = tqdm(total=ds_info.splits[split].num_examples,
                      desc="Merging %s" % split)

            for e in ds:
                x.append(e[self.conf.x_key])
                y.append(int(e[self.conf.y_key]))
                pb.update()
            pb.close()

        return x, y

    def resize_x(self, x):
        def resize(img, size):
            with tf.device("/cpu:0"):
                return tf.image.resize_with_pad(img, size, size)

        # x_resized = []
        for index, e in enumerate(tqdm(x, desc="resizing")):
            x[index] = resize(e, self.conf.img_size)

        return x

    def partition_dataset(self, x_resized, y):
        ds_index, ds_query = defaultdict(list), defaultdict(list)
        x_train, y_train, x_test, y_test = [], [], [], []
        train_cls = list(
            range(self.conf.train_classes[0], self.conf.train_classes[1]))
        test_cls = list(
            range(self.conf.test_classes[0], self.conf.test_classes[1]))

        for idx, e in enumerate(x_resized):
            cl = y[idx]
            if len(ds_index[cl]) < self.conf.index_shots:
                ds_index[cl].append(e)
            elif len(ds_query[cl]) < self.conf.query_shots:
                ds_query[cl].append(e)
            else:
                if cl in train_cls:
                    x_train.append(e)
                    y_train.append(cl)
                else:
                    x_test.append(e)
                    y_test.append(cl)

        return [
            ds_index, ds_query, x_train, y_train, x_test, y_test, train_cls,
            test_cls
        ]

    def flatten_indexes_queries(self, ds_index, ds_query, train_cls):
        # flatten the index
        x_index, y_index = [], []
        for k, es in ds_index.items():
            y_index.extend([k] * len(es))
            x_index.extend(es)

        # flatten query indexes
        x_unseen_queries, y_unseen_queries = [], []
        x_seen_queries, y_seen_queries = [], []
        for k, es in ds_query.items():
            if k in train_cls:
                y_seen_queries.extend([k] * len(es))
                x_seen_queries.extend(es)
            else:
                y_unseen_queries.extend([k] * len(es))
                x_unseen_queries.extend(es)

        return [
            x_index, y_index, x_unseen_queries, y_unseen_queries,
            x_seen_queries, y_seen_queries
        ]

    def save_dataset_tfrecord(self, x_train, y_train, x_test, y_test, x_index, y_index,
                     x_unseen_queries, y_unseen_queries, x_seen_queries,
                     y_seen_queries, train_cls, test_cls):
        fpath = "datasets/%s/%s/" % (self.version, self.dataset_name)
        cprint("Save files in %s" % fpath, "blue")
        clean_dir(fpath)

        files = [['train', x_train, y_train],
                ['test', x_test, y_test],
                ['index', x_index, y_index],
                ['unseen_queries', x_unseen_queries, y_unseen_queries],
                ['seen_queries', x_seen_queries, y_seen_queries]]

        def serialize_example(x, y):
            def _bytes_feature(value):
                """Returns a bytes_list from a string / byte."""
                if isinstance(value, type(tf.constant(0))):
                    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            def _float_feature(value):
                """Returns a float_list from a float / double."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

            def _int64_feature(value):
                """Returns an int64_list from a bool / enum / int / uint."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

            feature = {
                'x': _bytes_feature(tf.io.serialize_tensor(x)),
                'y': _int64_feature(y)
            }

            return tf.train.Example(features=tf.train.Features(feature=feature))

        for f in files:
            cprint("|-saving %s" % f[0], 'magenta')
            record_file = f'{fpath}{f[0]}.tfrecords'
            with tf.io.TFRecordWriter(record_file) as writer:
                for x, y in zip(f[1], f[2]):
                    tf_example = serialize_example(x, y)
                    writer.write(tf_example.SerializeToString())

    def save_dataset_sharded_npz(self, x_train, y_train, x_test, y_test, x_index, y_index,
                     x_unseen_queries, y_unseen_queries, x_seen_queries,
                     y_seen_queries, train_cls, test_cls):
        fpath = "datasets/%s/%s/" % (self.version, self.dataset_name)
        cprint("Save files in %s" % fpath, "blue")
        clean_dir(fpath)

        num_batches = 2
        partition_train = [
            i * len(x_train) // num_batches for i in range(1, num_batches + 1)
        ]
        look_back_train = len(x_train) // num_batches

        partition_test = [
            i * len(x_test) // num_batches for i in range(1, num_batches + 1)
        ]
        look_back_test = len(x_test) // num_batches

        partition_index = [
            i * len(x_index) // num_batches for i in range(1, num_batches + 1)
        ]
        look_back_index = len(x_index) // num_batches

        partition_unseen_queries = [
            i * len(x_unseen_queries) // num_batches
            for i in range(1, num_batches + 1)
        ]
        look_back_unseen_queries = len(x_unseen_queries) // num_batches

        partition_seen_queries = [
            i * len(x_seen_queries) // num_batches
            for i in range(1, num_batches + 1)
        ]
        look_back_seen_queries = len(x_seen_queries) // num_batches

        for shard_index, (train_i, test_i, index_i, unseen_queries_i, seen_queries_i) in enumerate(zip(
                partition_train, partition_test, partition_index,
                partition_unseen_queries, partition_seen_queries)):

            files = [
                [
                    'train', x_train[look_back_train:train_i],
                    y_train[look_back_train:train_i]
                ],
                [
                    'test', x_test[look_back_test:test_i],
                    y_test[look_back_test:test_i]
                ],
                [
                    'index', x_index[look_back_index:index_i],
                    y_index[look_back_index:index_i]
                ],
                [
                    'unseen_queries', x_unseen_queries[
                        look_back_unseen_queries:unseen_queries_i],
                    y_unseen_queries[look_back_unseen_queries:unseen_queries_i]
                ],
                [
                    'seen_queries',
                    x_seen_queries[look_back_seen_queries:seen_queries_i],
                    y_seen_queries[look_back_seen_queries:seen_queries_i]
                ]
            ]

            for f in files:
                cprint("|-saving %s" % f[0], 'magenta')
                fname = "%sshard_%s%s.npz" % (fpath, str(shard_index),  f[0])
                np.savez(fname, x=f[1], y=f[2])

    # Previous Method - Change config in build_dataset
    def save_dataset_whole_npz(self, x_train, y_train, x_test, y_test, x_index, y_index,
                     x_unseen_queries, y_unseen_queries, x_seen_queries,
                     y_seen_queries, train_cls, test_cls):
        fpath = "datasets/%s/%s/" % (self.version, self.dataset_name)
        cprint("Save files in %s" % fpath, "blue")
        clean_dir(fpath)

        files = [['train', x_train, y_train],
                ['test', x_test, y_test],
                ['index', x_index, y_index],
                ['unseen_queries', x_unseen_queries, y_unseen_queries],
                ['seen_queries', x_seen_queries, y_seen_queries]]
        for f in files:
            cprint("|-saving %s" % f[0], 'magenta')
            fname = "%s%s.npz" % (fpath, f[0])
            np.savez(fname, x=f[1], y=f[2])
    
    def conduct_sanity_checks(self, x_unseen_queries, y_unseen_queries,
                              x_seen_queries, y_seen_queries, x_index, y_index,
                              x_train, y_train, x_test, y_test, ds_index,
                              ds_query):
        assert len(y_unseen_queries) == len(x_unseen_queries)
        assert len(y_seen_queries) == len(x_seen_queries)
        assert len(y_index) == len(x_index)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        for lst in ds_index.values():
            assert len(lst) == self.conf.index_shots
        for lst in ds_query.values():
            assert len(lst) == self.conf.query_shots

        print(" |-train", len(x_train), len(y_train))
        print(" |-test", len(x_test), len(y_test))
        print(" |-index", len(x_index), len(y_index))
        print(" |-query seen", len(x_seen_queries), len(y_seen_queries))
        print(" |-query unseen", len(x_unseen_queries), len(y_unseen_queries))

    def build_dataset(self):
        cprint("[%s]\n" % self.dataset_name, 'yellow')

        # download and merge splits
        x, y = self.merge_dataset_splits()

        cprint("|-Resize", 'blue')
        x = self.resize_x(x)

        cprint("|-Partition", 'green')
        
        #TODO: prevent memory overload in smaller devices by creating sections one at a time
        [
            ds_index, ds_query, x_train, y_train, x_test, y_test, train_cls,
            test_cls
        ] = self.partition_dataset(x, y)

        [
            x_index, y_index, x_unseen_queries, y_unseen_queries,
            x_seen_queries, y_seen_queries
        ] = self.flatten_indexes_queries(ds_index, ds_query, train_cls)

        # sanity checks
        self.conduct_sanity_checks(x_unseen_queries, y_unseen_queries,
                                   x_seen_queries, y_seen_queries, x_index,
                                   y_index, x_train, y_train, x_test, y_test,
                                   ds_index, ds_query)

        save_method = "tfrecord" # tfrecord, full npz, sharded npz

        fpath = "datasets/%s/%s/" % (self.version, self.dataset_name)
        cprint("Save files in %s" % fpath, "blue")

        if save_method == "tfrecord":
            # save
            self.save_dataset_tfrecord(x_train, y_train, x_test, y_test, x_index, y_index,
                            x_unseen_queries, y_unseen_queries, x_seen_queries,
                            y_seen_queries, train_cls, test_cls)
        elif save_method == "full npz":
            # What previous generate_datasets.py did - do it if you want previous type of results
            self.save_dataset_whole_npz(x_train, y_train, x_test, y_test, x_index, y_index,
                            x_unseen_queries, y_unseen_queries, x_seen_queries,
                            y_seen_queries, train_cls, test_cls)
        elif save_method == "sharded_npz":
            self.save_dataset_sharded_npz(x_train, y_train, x_test, y_test, x_index, y_index,
                x_unseen_queries, y_unseen_queries, x_seen_queries,
                y_seen_queries, train_cls, test_cls)
        else:
            raise ValueError(f"Unknown save type: {save_method}")

        info = {
            "dataset": self.dataset_name,
            "splits": self.conf.splits,
            "img_size": self.conf.img_size,
            "num_classes": len(train_cls) + len(test_cls),
            "num_train_classes": len(train_cls),
            "num_test_classes": len(test_cls),
            "train_classes": train_cls,
            "test_classes": test_cls,
            "index_shots": self.conf.index_shots,
            "query_shots": self.conf.query_shots,
            "data": {
                "train": len(x_train),
                "test": len(x_test),
                "index": len(x_index),
                "unseen_queries": len(y_unseen_queries),
                "seen_queries": len(x_seen_queries)
            }
        }
        with open("%sinfo.json" % fpath, 'w') as o:
            o.write(json.dumps(info))

    def __call__(self):
        self.build_dataset()


def run(config):
    version = config['version']
    for dataset_name, dconf in config['datasets'].items():
        DatasetBuilder(dconf, dataset_name, version)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating datasets')
    parser.add_argument('--config', '-c', help='config path')
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
