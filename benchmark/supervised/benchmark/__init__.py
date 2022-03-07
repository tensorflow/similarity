"Shared functions"
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil


def _parse_image_function(example_proto):
    image_feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto,
                                         image_feature_description)

    parsed_image = tf.io.parse_tensor(example["x"], tf.float32)
    parsed_label = tf.cast(example['y'], tf.int32)

    return parsed_image, parsed_label


def load_tfrecord_unbatched_dataset(version, dataset_name, shard):
    path = "datasets/%s/%s/%s.tfrecords" % (version, dataset_name, shard)
    raw_dataset = tf.data.TFRecordDataset(path)
    return raw_dataset.map(_parse_image_function)

def load_tfrecord_dataset(version, dataset_name, shard, BATCH_SIZE):
    dataset = load_tfrecord_unbatched_dataset(version, dataset_name, shard)
    # dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

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
