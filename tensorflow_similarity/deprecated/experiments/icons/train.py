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

import multiprocessing
import random
import os
import pathlib
import subprocess
from tensorflow_similarity.layers.reduce_sum import ReduceSum
from tensorflow_similarity.layers.reduce_mean import ReduceMean
from tensorflow_similarity.layers.abs import Abs
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow.keras as keras
from absl import app, flags
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ProgbarLogger, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.python.client import device_lib

import tensorflow_similarity
from icongenerator.augmentation import LogoAugmentation
from icongenerator.generator import IconGenerator
from icongenerator.keras_generator import KerasGenerator
from icongenerator.visualization import NotebookCallback, grid

# Function used to specify hyper-parameters.
from kerastuner.distributions import Range, Choice, Boolean, Fixed

# Tuner used to search for the best model. Changing hypertuning algorithm
# simply requires to change the tuner class used.
from kerastuner.tuners import RandomSearch

import tensorflow as tf
from kerastuner.distributions import (Boolean, Choice, Fixed, Linear, Range,
                                      reset_distributions)
from kerastuner.tuners.randomsearch import RandomSearch
from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow_similarity.utils.model_utils import compute_size
from tensorflow_similarity.experiments.icons.constants import *
from tensorflow_similarity.experiments.icons.models import (
    flatten_model, simple_model, wdsr_model, resnet, mobilenet)
from tensorflow_similarity.experiments.icons.preprocessing import (
    mobilenet_preprocessor,
    batch_image_preprocessor,
    image_preprocessor,
    wdsr_image_preprocessor)
from tensorflow_similarity.api.callbacks.model_saver import MultiGPUSafeCheckpoint
from tensorflow_similarity.callbacks.validation_set_metrics import ValidationCallback
import signal
import faulthandler
faulthandler.register(signal.SIGUSR1)


flags.DEFINE_string("model", "wdsr", "")

FLAGS = flags.FLAGS


def decoder_model(embedding_input, feature_input):
    o = Dense(1024, activation="relu")(embedding_input)
    o = Dense(4096)(o)

    output_size = compute_size(feature_input.shape)

    o = Dense(output_size)(o)
    o = Reshape(feature_input.shape[1:])(o)

    reconstruction = Subtract()([o, feature_input])
    reconstruction = Abs()(o)
    reconstruction = ReduceMean()(o)

    return Model(
        inputs=[
            embedding_input,
            feature_input],
        outputs=[reconstruction])


def build_generator():
    return KerasGenerator(
        DATASET_FILENAME,
        append_clean_icon=False,
        icon_size=ICON_SIZE,
        autoencoder_icon_size=AUTOENCODER_ICON_SIZE,
        batch_size=BATCH_SIZE,
        examples_per_epoch=NUM_GEN_ICONS_PER_EPOCH,
        **AUGMENTATION_PARAMS)


def model(log_dir, tmp_dir):
    pp = image_preprocessor
    if FLAGS.model == "resnet":
        model = resnet()
    elif FLAGS.model == "wdsr":
        model = wdsr_model()
        pp = wdsr_image_preprocessor
    elif FLAGS.model == "flatten":
        model = flatten_model()
    elif FLAGS.model == "simple":
        model = simple_model()
    elif FLAGS.model == "mobilenet":
        model = mobilenet()
        pp = mobilenet_preprocessor
    else:
        raise ValueError("Bad --model")

    if NUM_GPUS > 1:
        model = multi_gpu_model(model, NUM_GPUS, cpu_merge=False)

    autoencoder = moirai.api.tasks.autoencoder.AutoencoderTask(
        "ae_image",
        model,
        decoder_model,
        ["anchor"],
        ["image"],
        loss_weight=.5,
        input_feature_type="augmented",
        target_feature_type="augmented")

    augmentation = LogoAugmentation(
        dataset_filename=DATASET_FILENAME,
        icon_size=ICON_SIZE,
        **AUGMENTATION_PARAMS)

    def denorm(x):
        return np.array(x * 255.0, dtype=np.uint8)

    similarity_learner = SimHash(model,
                                 preprocessing=pp,
                                 augmentation=augmentation,
                                 #                                 auxillary_tasks=[autoencoder],
                                 tmp_dir=tmp_dir,
                                 batch_size=BATCH_SIZE,
                                 log_dir=log_dir,
                                 visualize_img_fields=["image"],
                                 visualize_img_denormalization=[denorm],
                                 step_size_multiplier=FLAGS.step_size_multiplier,
                                 strategy="hard_quadruplet_loss")
    return similarity_learner


class SimilarityDisplayCallback(keras.callbacks.Callback):

    def __init__(self, x_test, y_test, x_targets, y_targets):
        # Un-normalize the images (which get_augmented_validation_data returns
        # as an array where elements are between 0.0 and 1.0)
        self.x_test = x_test
        self.y_test = y_test

        self.x_target = x_targets
        self.y_target = y_targets

        self.simhash_interface = None

    def set_simhash(self, simhash):
        self.simhash_interface = simhash

    def save_to_disk(self, category, variant, epoch, image_id, image):
        directory = 'predictions/%s' % MODEL_ID
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        path = '%s/%s-%s-icon_%d-epoch_%d.png' % (
            directory, category, variant, image_id, epoch)
        Image.fromarray(image, 'RGB').save(path)

    def _neighbor_map(self, epoch):
        e_test = self.simhash_interface.predict(self.x_test)

        db = self.simhash_interface.build_database(
            self.x_target, self.y_target)
        neighbors = db.query(e_test, N=len(self.y_target))

        # Build a rank-map image, where each row represents one item in the validation set, and the
        # the highlighted column represents the rank of the true class in the
        # predictions.
        img = Image.new("RGB", (len(self.x_test), len(
            self.x_target)), color=(255, 255, 0))
        pixels = img.load()
        violet = (238, 130, 238)

        for item_idx, (neighbor_list, item_class) in enumerate(
                zip(neighbors, self.y_test)):
            for neighbor_rank, neighbor in enumerate(neighbor_list):
                if neighbor.index == item_class:
                    pixels[neighbor_rank, item_idx] = violet

        path = os.path.join(MODEL_ID, 'epoch_%05d.png' % epoch)
        img = img.resize((len(self.x_test) * 5, len(self.x_target) * 5))

        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Epoch %d" % epoch, fill=(0, 0, 0, 255))
        img.save(path)
        return img

    def on_epoch_end(self, epoch, logs={}):
        cleaned = {}
        for k, v in logs.items():
            if "pseudometric" not in k:
                cleaned[k] = v
        self._neighbor_map(epoch)


def gen_batch(x):
    np.random.seed(os.getpid())
    random.seed(os.getpid())
    generator = build_generator()
    x, y = generator.get_augmented_validation_data(
        normalize=False)
    return x, [x.decode("utf-8") for x in generator.dataset.logo_labels]


def create_metrics_callback(generator):
    x_targets, y_targets = generator.get_clean_validation_data(
        normalize=False)

    labels = [label.decode("utf-8") for label in generator.dataset.logo_labels]

    x_test = []
    y_test = []

    tasks = [x for x in range(100)]

    pool = multiprocessing.Pool(50)
    for idx, (x, y) in enumerate(pool.imap_unordered(gen_batch, tasks)):
        print("Generated test set (%d/%d)" % (idx + 1, 100))
        x_test.extend(x)
        y_test.extend(y)

    x_val = {"targets": {"image": []}}
    y_val = {"targets": []}

    for logo, label in zip(x_targets, labels):
        x_val["targets"]["image"].append(logo)
        y_val["targets"].append(label)

    for logo, label in zip(x_test, y_test):
        if label not in x_val:
            x_val[label] = {"image": []}
            y_val[label] = []
        x_val[label]["image"].append(logo)
        y_val[label].append(label)

    for label in x_val.keys():
        x_val[label]["image"] = np.array(x_val[label]["image"])
        y_val[label] = np.array(y_val[label])

    return ValidationCallback(x_val, y_val)


def main(args):
    tmp_dir = os.path.join("results", MODEL_ID, "tmp")
    log_dir = os.path.join("results", MODEL_ID, "logs")
    tf.io.gfile.makedirs(log_dir)
    tf.io.gfile.makedirs(tmp_dir)

    if tf.io.gfile.exists("/tmp/blackboard.data"):
        tf.io.gfile.remove("/tmp/blackboard.data")

    generator = build_generator()
    print("Starting model: %s." % MODEL_ID)
    print("Random Accuracy would be: %5.4f." % (1 / (generator.num_classes)))
    print("Num classes: %d." % generator.num_classes)

    saver_callback = MultiGPUSafeCheckpoint(
        output_dir=os.path.join("results", MODEL_ID))
    metrics_callback = create_metrics_callback(generator)
    tensorboard = TensorBoard(
        log_dir=os.path.join("results", MODEL_ID, "logs"),
        write_images=True,
        embeddings_freq=0,
        embeddings_layer_names=[])

    ICONS = [
        "imgur",
        "twitter",
        "google",
        "gmail",
        "facebook",
        "twitter",
        "linkedin",
        "google_youtube",
        "amazon",
        "paypal"]

    x = []
    y = []
    for ico, label in zip(generator.dataset.logos,
                          generator.dataset.logo_labels):
        label = label.decode("utf-8")
        if label in ICONS:
            x.append(np.array(ico))
            y.append(label)

    x = {"image": np.array(x)}
    y = np.array(y)

    simhash = model(log_dir, tmp_dir)
    simhash.fit(x,
                y,
                prewarm_epochs=0,
                generator_workers=60,
                callbacks=[
                    saver_callback,
                    metrics_callback,
                    tensorboard
                ])


if __name__ == '__main__':
    app.run(main)
