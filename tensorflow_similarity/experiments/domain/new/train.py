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

# Register a signal handler that listens to USR1, and dumps a stack trace
# of the current threads when such a signal is received.
import faulthandler
import gzip
import sys
import json
import os
import signal
import struct
import subprocess
import time

import h5py
import numpy as np
import six
import tensorflow as tf
from absl import app, flags
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Layer, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import kerastuner
import tensorflow_similarity
import tabulate
import tensorflow_tensortext as tensortext
from kerastuner.distributions import Boolean, Choice, Fixed, Range
from kerastuner.tuners import RandomSearch
from tensorflow_similarity.api.callbacks.model_saver import MultiGPUSafeCheckpoint
from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow_similarity.callbacks.base import MoiraiCallback
from tensorflow_similarity.callbacks.hard_mining import ResultWriter
from tensorflow_similarity.callbacks.validation_set_metrics import ValidationCallback
from tensorflow_similarity.dataset import Dataset, DatasetConfig, Transformation
from tensorflow_tensortext.layers import CharSeq
from tensorflow_similarity.experiments.domain.new.augment import DomainAugment
from tensorflow_similarity.experiments.domain.new.models import *
from tensorflow_similarity.readers.inmemory import MemoryReader
from tensorflow_similarity.utils.config_utils import (get_module_version,
                                                      register_custom_object)
from tensorflow.python import ConfigProto, Session
from tensorflow_similarity.utils.model_utils import compute_size
from tensorflow_similarity.layers.reduce_sum import ReduceSum
from tensorflow_similarity.layers.reduce_mean import ReduceMean
from tensorflow_similarity.layers.abs import Abs
from tensorflow_tensortext.engine.hashing import char2seq
from tensorflow_similarity.api.filters.text_distance import EditDistanceFilter

import traceback
faulthandler.register(signal.SIGUSR1)

flags.DEFINE_string("strategy", "stable_hard_quadruplet_loss", "")

flags.DEFINE_string("base_data_dir", os.path.join(".", "data"),
                    "Path to the domain data.")
flags.DEFINE_float("sample_fraction", .1,
                   "Fraction of the non-popular domains to sample.")
flags.DEFINE_float("learning_rate", .0001, "Starting learning rate.")
flags.DEFINE_string("model", "trigrams", "")
flags.DEFINE_string("checkpoint_file", None, "")
flags.DEFINE_integer("patience", 20, "")
flags.DEFINE_string("base_output_dir", ".",
                    "Main directory where results are stored.")
flags.DEFINE_string(
    "api_key", None,
    "API tuner from: https://keras-tuner.appspot.com/getstarted")
flags.DEFINE_boolean("hypertune", False, "Whether to use the tuner.")
flags.DEFINE_integer("epochs_per_model", 3, "")
flags.DEFINE_integer("epoch_budget", 100, "")
flags.DEFINE_boolean("debug_memory", False, "")

FLAGS = flags.FLAGS


class OProxy(object):

    def flush(self, *args, **kwargs):
        sys.__stdout__.flush(*args, **kwargs)

    def write(self, *args, **kwargs):
        traceback.print_stack()
        return sys.__stdout__.write(*args, **kwargs)


# print(dir(sys.stdout))
#sys.stdout = OProxy()


def trigrams():
    return kim_char_cnn_ngrams(
        input_size=32,
        max_grams=3,
        alphabet_size=69,
        tensortext_decoder_layers=0,
        tensortext_decoder_layer_size=64,
        conv_layers=[[256, 10], [256, 7], [256, 5], [256, 3]],
        fully_connected_layers=[512, 512, 256],
        dropout_p=.1,
        embedding_size=32,
        banks=6,
        bits=4,
        optimizer='adam',
        loss='categorical_crossentropy')


"""
┌──────────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ hyperparam           │ model 0 │ model 1 │ model 2 │ model 3 │ model 4 │ model 5 │ model 6 │ model 7 │ model 8 │ model 9 │
├──────────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ default              │         │         │         │         │         │         │         │         │         │         │
│ |-banks              │ 6       │ 4       │ 4       │ 4       │ 4       │ 6       │ 6       │ 2       │ 6       │ 6       │
│ |-bits               │ 4       │ 8       │ 8       │ 8       │ 6       │ 4       │ 4       │ 8       │ 8       │ 4       │
│ |-decoder_layer_size │ 64      │ 8       │ 8       │ 32      │ 8       │ 32      │ 64      │ 16      │ 64      │ 64      │
│ |-decoder_layers     │ 0       │ 1       │ 2       │ 1       │ 1       │ 2       │ 2       │ 0       │ 2       │ 0       │
│ |-embedding_size     │ 32      │ 128     │ 64      │ 32      │ 128     │ 64      │ 48      │ 128     │ 64      │ 128     │
│ |-fc_layers          │ 1       │ 1       │ 1       │ 1       │ 1       │ 1       │ 1       │ 2       │ 1       │ 1       │
└──────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
"""


def hyper_trigrams():
    decoder_layers = Choice("decoder_layers", [0, 1, 2])
    decoder_layer_size = Choice("decoder_layer_size", [8, 16, 32, 64])
    fully_connected_layers = Range("fc_layers", 1, 3)
    bits = Choice("bits", [4, 6, 8])
    banks = Choice("banks", [2, 4, 6, 8, 10, 12])
    embedding_size = Choice("embedding_size", [32, 48, 64, 128])

    fully_connected_layers = [512 for _ in range(fully_connected_layers)]
    fully_connected_layers[-1] = 256

    return kim_char_cnn_ngrams(
        input_size=32,
        max_grams=3,
        alphabet_size=69,
        tensortext_decoder_layers=decoder_layers,
        tensortext_decoder_layer_size=decoder_layer_size,
        conv_layers=[[256, 10], [256, 7], [256, 5], [256, 3]],
        fully_connected_layers=fully_connected_layers,
        dropout_p=.1,
        embedding_size=embedding_size,
        banks=banks,
        bits=bits,
        optimizer='adam',
        loss='categorical_crossentropy')


def bigrams():
    return kim_char_cnn_ngrams(
        input_size=32,
        max_grams=2,
        alphabet_size=69,
        tensortext_decoder_layers=1,
        tensortext_decoder_layer_size=64,
        conv_layers=[[256, 10], [256, 7], [256, 5], [256, 3]],
        fully_connected_layers=[512, 512, 256],
        dropout_p=.1,
        embedding_size=32,
        banks=2,
        bits=8,
        optimizer='adam',
        loss='categorical_crossentropy')


def read_file(filename):
    full_filename = os.path.join(FLAGS.base_data_dir, filename)

    print("Reading: ", full_filename)
    with tf.io.gfile.GFile(full_filename, 'r') as f:
        lines = f.read().split("\n")
        return [line.split(",")[-1] for line in lines]


def read_training_data():
    x_train = read_file("domains.popular")
    x_test = []

    for file in ["domains.tail"]:
        x_ = read_file(file)
        np.random.shuffle(x_)
        train_size = int(len(x_) * FLAGS.sample_fraction)

        x_train.extend(x_[:train_size])

    tmp = []
    for domain in tqdm(list(set(x_train)), desc="Filtering by size."):
        if len(domain) > 2 and len(domain) <= 32:
            tmp.append(domain)
    x_train = tmp

    print("Reading goldenset.")
    goldenset = os.path.join(FLAGS.base_data_dir, "golden_set_5")

    with h5py.File(goldenset) as f:
        x_val = f['examples']
        y_val = f['labels']
        g_val = f['groups']

        x_test = {}
        y_test = {}

        for x, y, group in zip(x_val, y_val, g_val):
            if group not in x_test:
                x_test[group] = {"example": []}
                y_test[group] = []
            x_test[group]["example"].append(x)
            y_test[group].append(y)

        for group in x_test.keys():
            x_test[group]["example"] = np.array(x_test[group]["example"])

    y_train = np.array([i for (i, v) in enumerate(x_train)])

    x_train = {"example": np.array(x_train)}
    return (x_train, y_train), (x_test, y_test)


class DecoderTargetPreprocessing(object):
    def __init__(self):
        self.model = self._model()

    def __call__(self, example):
        return self.model.predict([np.array([example])])[0]

    def _model(self):
        i = Input(shape=(1,), dtype=tf.string, name="input")
        o = CharSeq(sequence_length=32,
                    input_bits=8,
                    banks=8,
                    bits_per_bank=8,
                    dropout=0.0,
                    center_results=True)(i)
        m = Model(i, o)
        m.compile(loss="mse", optimizer="adam")
        return m


def decoder_model(embedding_tensor, input_tensor):
    o = Dense(1024, activation="relu")(embedding_tensor)
    o = Dense(2048)(o)
    o = Reshape((32, 64))(o)

    # Since we're using a string input, we need to transform
    # the raw input into a float sequence.
    tt = CharSeq(
        input_bits=8,
        sequence_length=32,
        banks=8,
        bits_per_bank=8,
        dropout=0.0,
        center_results=True)(input_tensor)

    reconstruction = Subtract()([tt, o])
    reconstruction = Abs()(reconstruction)
    reconstruction = ReduceMean()(reconstruction)
    return Model(
        inputs=[embedding_tensor, input_tensor],
        outputs=[reconstruction])


def train(data, model, learning_rate=.0001):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    session = Session(config=config)
    tf.keras.backend.set_session(session)

    (x_train, y_train), (x_test, y_test) = data
    if os.path.exists("blackboard.data"):
        os.unlink("blackboard.data")

    print("Creating SimHash metamodel with strategy: ", FLAGS.strategy)

    autoencoder = moirai.api.tasks.autoencoder.AutoencoderTask(
        "ae_example",
        model,
        decoder_model,
        ["anchor"],
        ["example"],
        loss_weight=.01,
        input_feature_type="preprocessed",
        target_feature_type="preprocessed",
        target_preprocessing=DecoderTargetPreprocessing())

    simhash = SimHash(
        model,
        auxillary_tasks=[autoencoder],
        pair_filter=EditDistanceFilter(
            feature_name="example",
            min_length=5,
            min_distance=5,
            max_distance=10),
        augmentation=DomainAugment(2),
        log_dir="/tmp/testrun",
        visualize_text_fields=["example"],
        optimizer=Adam(
            lr=learning_rate),
        strategy=FLAGS.strategy,
        step_size_multiplier=FLAGS.step_size_multiplier)

    timestr = time.strftime("%Y%m%d_%H%M%S")

    output_directory = "%s_%s_%s" % (timestr, FLAGS.strategy, FLAGS.model)
    output_directory = os.path.join(FLAGS.base_output_dir, output_directory)
    os.makedirs(output_directory)

    with tf.io.gfile.GFile(os.path.join(output_directory, "METADATA"), "w") as f:
        f.write("Flags:\n")
        for k, v in six.iteritems(FLAGS.flag_values_dict()):
            f.write("  %s=%s" % (k, v))

    if FLAGS.checkpoint_file:
        if os.path.exists(FLAGS.checkpoint_file):
            import shutil
            shutil.copy2(
                FLAGS.checkpoint_file,
                os.path.join(output_directory, "warm_start_checkpoint"))
            model = tf.keras.models.load_model(FLAGS.checkpoint_file)
            weights = model.get_weights()
            simhash.set_weights(weights)

    simhash.fit(
        x_train,
        y_train,
        prewarm_epochs=3,
        epochs=FLAGS.epochs,
        callbacks=[
            ValidationCallback(
                x_test, y_test, neighborhood_workers=0, colorize=True),
            ReduceLROnPlateau(monitor='loss', verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                mode='min',
                min_delta=0.0001,
                patience=FLAGS.patience),
            MultiGPUSafeCheckpoint(
                output_dir=".",
                tmp_dir=None,
                filename="keras_checkpoint.{epoch:03d}-{loss:.5f}.hdf5")
        ],
        generator_workers=60
    )

    return simhash


WRAPPED_MODEL_COUNT = 0


def main(args):

    data = read_training_data()

    if FLAGS.model == "ensemble":
        model = ensemble_model
    elif FLAGS.model == "trigrams":
        model = trigrams
    elif FLAGS.model == "hypertrigrams":
        model = hyper_trigrams
    elif FLAGS.model == "bigrams":
        model = bigrams
    elif FLAGS.model == "dense":
        model = dense
    else:
        print("Unknown model: ", FLAGS.model)
        return

    m = train(data, model(), learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    app.run(main)
