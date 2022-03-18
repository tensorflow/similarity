"Shared functions"
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
import os

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

    return parsed_image / 255, parsed_label


def ds_get_cardinality(ds):
    i = 0
    for _ in ds:
        i += 1
    return ds.apply(tf.data.experimental.assert_cardinality(i))

def load_tfrecord_unbatched_dataset(version, dataset_name, shard):
    # print(os.listdir())
    
    path = f"../../../datasets/{version}/{dataset_name}/{shard}.tfrecords"
    full_path = os.path.join(os.path.dirname(__file__), path)

    raw_dataset = tf.data.TFRecordDataset(full_path)
    return ds_get_cardinality(raw_dataset.map(_parse_image_function))

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

def get_gpu_availability():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def clean_dir(fpath):
    "delete previous content and recreate dir"
    dpath = Path(fpath)
    if dpath.exists():
        shutil.rmtree(fpath)
    dpath = dpath.mkdir(parents=True)
