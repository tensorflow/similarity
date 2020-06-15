# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import collections
import faulthandler
import json
import math
import multiprocessing
import pdb
import signal

import numpy as np
import scipy
import six
import sklearn
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

import tensorflow_similarity
from tensorflow_similarity.bundle.bundle_lib import BundleWrapper
from tensorflow_similarity.dataset import Dataset, DatasetConfig
from tensorflow_similarity.api.engine.inference import InferenceRequest
from tensorflow_similarity.api.engine.nearest_neighbors import \
    get_best_available_nearest_neighbor_algorithm
from tensorflow_similarity.utils.config_utils import (
    deserialize_moirai_object,
    json_dict_to_moirai_obj,
    load_custom_configuration,
    load_json_configuration_file,
    serialize_moirai_object)

faulthandler.register(signal.SIGUSR1)


Embedding = collections.namedtuple("Embedding", ["point", "label", "label_id"])


def predict_feature_array(inferer, features, request=InferenceRequest()):
    return [x['embedding'] for x in inferer.embed(features, request=request)]


def get_dataset(config, reader_name, dataset_cfg_name):
    reader = json_dict_to_moirai_obj(config[reader_name])
    common_dataset_config = config.get('common_dataset', None)
    dataset_config = config.get(dataset_cfg_name, common_dataset_config)
    dataset_config = json_dict_to_moirai_obj(dataset_config)
    data = reader.read()
    dataset = Dataset(data, dataset_config)
    return dataset


def embed_dataset(inference, dataset, augment=False):
    examples = []
    labels = []
    ids = []
    predictions = []

    size = len(dataset)

    for i in tqdm(range(size), total=size, desc="Preparing examples."):
        examples.append(dataset.example(
            i, augment=augment, preprocess=True))
        labels.append(dataset.label(i))
        ids.append(dataset.id(i))

    request = InferenceRequest(progress_bar=True)
    predictions = predict_feature_array(inference, examples, request)

    data = []
    pbar = tqdm(total=len(labels), desc="Converting to Embedding objects")
    for label, id, prediction in zip(labels, ids, predictions):
        data.append(Embedding(point=prediction, label=label, label_id=id))
        pbar.update(1)
    pbar.close()
    return data


InferenceTask = collections.namedtuple("InferenceTask",
                                       ["examples", "labels", "augment"])


def generate_embeddings_for_calibration(config, bundle, target_augments=5):

    tqdm.write("  Reading targets dataset...")
    targets_ds = get_dataset(config, "targets_reader", "targets_dataset")
    targets_size = len(targets_ds)
    tqdm.write("  Reading examples dataset...")
    inference_ds = get_dataset(
        config, "calibration_reader", "calibration_dataset")
    inference_size = len(inference_ds)

    inference = bundle.get_inference()

    tqdm.write("Embedding non targets with no augmentation.")
    non_targets = embed_dataset(inference, inference_ds, augment=False)

    tqdm.write("Embedding raw targets with no augmentation.")
    targets = embed_dataset(inference, targets_ds, augment=False)

    augmented_targets = []
    for i in range(target_augments):
        tqdm.write("Generating augmented targets (%d/%d)" %
                   (i, target_augments))
        augmented_targets.extend(embed_dataset(
            inference, targets_ds, augment=True))

    return {
        "raw_targets": targets,
        "augmented_targets": augmented_targets,
        "other_points": non_targets
    }


def load_datafile(filename):
    labels = []
    embs = []
    ids = []

    with tf.io.gfile.GFile(filename, "r") as f:
        tqdm.write("Reading ", filename)
        lines = f.read().split("\n")
        for line in tqdm(lines, desc="Parsing."):
            try:
                label, id, emb_str = line.split("\t")
                emb = np.array([float(x) for x in emb_str.split(",")])
                labels.append(label)
                embs.append(emb)
                ids.append(id)
            except BaseException:
                tqdm.write("Failed to parse: %s" % line)

    return labels, ids, embs


def compute_target_predictions(task):
    target_points, target_labels, embeddings = task

    for embedding in embeddings:
        result = nn_lookup.query_one(np.asarray(embedding.point), k=50)
        for neighbor in result:
            predicted_label = target_labels[neighbor.index]
            target_label = embedding.label

            y_trues.append(target_label)
            y_preds.append(predicted_label)
            distances.append(neighbor.distance)

    return (y_trues, y_preds, distances)


def compute_predictions(task):
    nn_class = get_best_available_nearest_neighbor_algorithm()

    target_points, target_labels, embeddings = task

    nn_lookup = nn_class(target_points)

    y_trues = []
    y_preds = []
    distances = []

    for embedding in embeddings:
        result = nn_lookup.query_one(np.asarray(embedding.point), k=50)
        for neighbor in result:
            predicted_label = target_labels[neighbor.index]
            target_label = embedding.label

            y_trues.append(target_label)
            y_preds.append(predicted_label)
            distances.append(neighbor.distance)

    return (y_trues, y_preds, distances)


def convert_to_binary_classification_problem(y_trues, y_preds, distances):
    y_true_out = []
    y_pred_out = []

    for y_true, y_pred, d in zip(y_trues, y_preds, distances):
        y_true_out.append(1 if y_true == y_pred else 0)
        y_pred_out.append(-d)

    return y_true_out, y_pred_out


def calibrate(config, bundle):
    nn_class = get_best_available_nearest_neighbor_algorithm()

    embeddings = generate_embeddings_for_calibration(config, bundle)

    raw_targets = embeddings["raw_targets"]
    aug_targets = embeddings["augmented_targets"]
    other_points = embeddings["other_points"]

    nn_points = []
    nn_labels = []

    for embedding in raw_targets:
        nn_points.append(embedding.point)
        nn_labels.append(embedding.label)

    nn = nn_class(nn_points)

    label_to_point = {}
    for emb in raw_targets:
        label_to_point[emb.label] = emb.point

    positives = []
    for emb in aug_targets:
        real = label_to_point[emb.label]
        positives.append(np.linalg.norm(
            np.asarray(real) - np.asarray(emb.point), ord=2))

    chunks = np.array_split(other_points, 256)

    tasks = []
    for chunk in chunks:
        embs = []
        for item in chunk:
            embs.append(
                Embedding(point=item[0], label=item[1], label_id=item[2]))
        tasks.append((nn_points, nn_labels, embs))

    pool = multiprocessing.Pool(64)

    y_trues = []
    y_preds = []
    distances = []
    pbar = tqdm(total=len(tasks), desc="Finding neighbors.")
    for y_trues_, y_preds_, distances_ in pool.imap_unordered(
            compute_predictions, tasks):
        pbar.update(1)
        y_trues.extend(y_trues_)
        y_preds.extend(y_preds_)
        distances.extend(distances_)
    pbar.close()

    y_true = ([1] * len(positives)) + ([0] * len(distances))
    y_pred_real = np.array([x for x in (positives + distances)])

    y_pred = np.array([-x for x in (positives + distances)])

    y_max = np.max(y_pred)
    y_min = np.min(y_pred)

    offset = -y_min
    y_pred += offset
    y_pred /= (y_max + offset)

    precisions, recalls, shifted_thresholds = sklearn.metrics.precision_recall_curve(
        y_true, y_pred)
    precisions = precisions[:-1].tolist()
    recalls = recalls[:-1].tolist()

    thresholds = shifted_thresholds
    thresholds *= (y_max + offset)
    thresholds -= offset
    thresholds = -thresholds

    precisions.append(1.01)
    recalls.append(0)
    thresholds.tolist().append(-1)

    thresholds_for_bundle = []
    with open("prt_curve.csv", "wt") as f:
        last = -1
        for p, r, t in zip(precisions, recalls, thresholds):
            rounded_t = round(t, 3)
            if rounded_t != last:
                last = rounded_t
                thresholds_for_bundle.append([t, p])
                f.write("%s, %s, %s\n" % (p, r, t))
                tqdm.write("P=%4.2f R=%4.2f @ Threshold %4.2f" % (p, r, t))

    bundle.set_global_thresholds(thresholds_for_bundle)
