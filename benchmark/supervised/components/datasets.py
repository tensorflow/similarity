import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

from tensorflow_similarity.samplers import MultiShotMemorySampler


def load_tf_dataset(dataset_name, cfg, preproc_fns):
    x, y = [], []
    split = "all"
    ds, ds_info = tfds.load(dataset_name, split=split, with_info=True)

    if cfg["x_key"] not in ds_info.features:
        raise ValueError("x_key not found - available features are:", str(ds_info.features.keys()))
    if cfg["y_key"] not in ds_info.features:
        raise ValueError("y_key not found - available features are:", str(ds_info.features.keys()))

    pb = tqdm(total=ds_info.splits[split].num_examples, desc="converting %s" % split)

    # unpack the feature and labels
    for e in ds:
        x.append(e[cfg["x_key"]])
        y.append(e[cfg["y_key"]])
        pb.update()
    pb.close()

    # Apply preproccessing
    with tf.device("/cpu:0"):
        for idx in tqdm(range(len(x)), desc="Preprocessing data"):
            for p in preproc_fns:
                x[idx] = p(x[idx])

    return x, y


def create_splits(x, y, cfg, fold_id):
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
    val_start = val_len * fold_id
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


def make_sampler(x, y, tconf, aug_fns):
    def augmentation_fn(x, y, *args):
        for a in aug_fns:
            x = a(x)
        return x, y

    return MultiShotMemorySampler(
        x,
        y,
        classes_per_batch=tconf.get("classes_per_batch", 2),
        examples_per_class_per_batch=tconf.get("examples_per_class_per_batch", 2),
        augmenter=augmentation_fn,
    )


def make_eval_data(x, y, aug_fns):
    with tf.device("/cpu:0"):
        for idx in tqdm(range(len(x)), desc="Preprocessing data"):
            for p in aug_fns:
                x[idx] = p(x[idx])

    unique, counts = np.unique(y, return_counts=True)
    class_counts = {k: v for k, v in zip(unique, counts)}

    return (tf.convert_to_tensor(np.array(x)), tf.convert_to_tensor(np.array(y)), class_counts)
