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

from benchmark import clean_dir

tf.config.set_visible_devices([], 'GPU')
cprint('Tensorflow set to CPU', 'green')


def resize(img, size):
    with tf.device("/cpu:0"):
        return tf.image.resize_with_pad(img, size, size)


def run(config):
    version = config['version']
    for dataset_name, dconf in config['datasets'].items():
        cprint("[%s]\n" % dataset_name, 'yellow')

        # conf
        x_key = dconf['x_key']
        y_key = dconf['y_key']
        splits = dconf['splits']
        train_classes = dconf['train_classes']
        test_classes = dconf['test_classes']
        img_size = dconf['shape'][0]
        index_shots = dconf['index_shots']
        query_shots = dconf['query_shots']

        # download and merge splits
        x = []
        y = []
        for split in splits:
            ds, ds_info = tfds.load(dataset_name, split=split, with_info=True)

            if x_key not in ds_info.features:
                raise ValueError("x_key not found - available features are:",
                                 str(ds_info.features.keys()))
            if y_key not in ds_info.features:
                raise ValueError("y_key not found - available features are:",
                                 str(ds_info.features.keys()))

            pb = tqdm(total=ds_info.splits[split].num_examples,
                      desc="Merging %s" % split)

            for e in ds:
                x.append(e[x_key])
                y.append(int(e[y_key]))
                pb.update()
            pb.close()

    cprint("|-Resize", 'blue')
    x_resized = []
    for e in tqdm(x, desc="resizing"):
        x_resized.append(resize(e, img_size))

    cprint("|-Partition", 'green')
    ds_index = defaultdict(list)
    ds_query = defaultdict(list)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_cls = list(range(train_classes[0], train_classes[1]))
    test_cls = list(range(test_classes[0], test_classes[1]))

    for idx, e in enumerate(x_resized):
        cl = y[idx]
        if len(ds_index[cl]) < index_shots:
            ds_index[cl].append(e)
        elif len(ds_query[cl]) < query_shots:
            ds_query[cl].append(e)
        else:
            if cl in train_cls:
                x_train.append(e)
                y_train.append(cl)
            else:
                x_test.append(e)
                y_test.append(cl)

    # flatten the index
    x_index = []
    y_index = []
    for k, es in ds_index.items():
        y_index.extend([k] * len(es))
        x_index.extend(es)

    # flatten query indexes
    x_unseen_queries = []
    y_unseen_queries = []
    x_seen_queries = []
    y_seen_queries = []
    for k, es in ds_query.items():
        if k in train_cls:
            y_seen_queries.extend([k] * len(es))
            x_seen_queries.extend(es)
        else:
            y_unseen_queries.extend([k] * len(es))
            x_unseen_queries.extend(es)

    # sanity checks
    assert len(y_unseen_queries) == len(x_unseen_queries)
    assert len(y_seen_queries) == len(x_seen_queries)
    assert len(y_index) == len(x_index)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    for lst in ds_index.values():
        assert len(lst) == index_shots
    for lst in ds_query.values():
        assert len(lst) == query_shots

    print(" |-train", len(x_train), len(y_train))
    print(" |-test", len(x_test), len(y_test))
    print(" |-index", len(x_index), len(y_index))
    print(" |-query seen", len(x_seen_queries), len(y_seen_queries))
    print(" |-query unseen", len(x_unseen_queries), len(y_unseen_queries))

    # save
    fpath = "datasets/%s/%s/" % (version, dataset_name)
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
    info = {
        "dataset": dataset_name,
        "splits": splits,
        "img_size": img_size,
        "num_classes": len(train_cls) + len(test_cls),
        "num_train_classes": len(train_cls),
        "num_test_classes": len(test_cls),
        "train_classes": train_cls,
        "test_classes": test_cls,
        "index_shots": index_shots,
        "query_shots": query_shots,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating datasets')
    parser.add_argument('--config', '-c', help='config path')
    args = parser.parse_args()

    if not args.config:
        parser.print_usage()
        quit()
    config = json.loads(open(args.config).read())
    run(config)
