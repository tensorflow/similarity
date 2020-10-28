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

from absl import app, flags
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import json
import tensorflow_similarity
import os
import time
from tensorflow_similarity import trainer
from tensorflow_similarity.experiments.domain import domain_augmentation
from tensorflow_similarity.filters.text_distance import EditDistanceFilter
from tensorflow_similarity.utils.config_utils import load_custom_configuration, json_dict_to_moirai_obj
import sys
import tensorflow as tf

flags.DEFINE_string(
    'model', None, 'Name of the model (defined in moirai/architectures.py)')
flags.DEFINE_enum(
    'strategy', 'batch_triplets', [
        'batch_triplets', 'batch_quadruplets', 'online_triplets',
        'online_triplets_hard', "hard_quads"
    ], 'Name of the strategy to use for training.'
    'Name of the model (defined in moirai/architectures.py)')
flags.DEFINE_string(
    'config', None, 'Moirai configuration file specifying the'
    'inputs, validation sets, etc.')

FLAGS = flags.FLAGS


def get_optional_config(config, name):
    if name not in config:
        return None
    cfg = config[name]
    return json_dict_to_moirai_obj(cfg)


def moirai_20180926(input_shape):
    """5 layer dense, stacked on an Bi-di GRU """
    input_seq = Input(shape=input_shape['example'], name='example')

    rnn = Bidirectional(GRU(32, return_sequences=True, unroll=True))(input_seq)
    flattened = Flatten()(rnn)
    dense = Dense(128, activation="relu")(flattened)
    dense = Dense(128, activation="relu")(dense)
    dense = Dense(128, activation="relu")(dense)
    dense = Dense(128, activation="relu")(dense)
    output = Dense(128, activation="selu")(dense)

    model = Model(input_seq, output)
    model.compile(optimizer="adam", loss='mse')
    return model


def main(argv):
    load_custom_configuration()

    tf.io.gfile.MakeDirs(FLAGS.output_dir)
    tf.io.gfile.MakeDirs(FLAGS.local_tmp_dir)

    with tf.io.gfile.GFile(FLAGS.config, "r") as f:
        config = f.read()
        config = json.loads(config)

    reader = json_dict_to_moirai_obj(config['input'])
    dataset = json_dict_to_moirai_obj(config['dataset'])
    augmentation = get_optional_config(config, "augmentation")
    augmented_dataset = get_optional_config(config, 'augmented_dataset')
    filter = get_optional_config(config, 'filter')
    validation_reader = get_optional_config(config, 'validation_reader')
    validation_dataset = get_optional_config(config, 'validation_dataset')

    print("Strategy: %s" % FLAGS.strategy)

    trainer.train(
        reader=reader,
        dataset=dataset,
        filter=filter,
        augmented_dataset=augmented_dataset,
        augmentation=augmentation,
        validation_reader=validation_reader,
        validation_dataset=validation_dataset,
        epochs=FLAGS.epochs,
        model_name=FLAGS.model,
        num_gpus=FLAGS.num_gpus,
        strategy_name=FLAGS.strategy)


def jupyter_main(model,
                 strategy,
                 epochs,
                 config,
                 output_dir,
                 tmp_dir,
                 dependency_module="moirai.experiments.domain.dependencies"):
    sys.argv = [
        "python",
        "--model=jupyter_model",
        "--strategy=%s" % strategy,
        "--custom_dependency_module=%s" % dependency_module,
        "--output_dir=%s" % output_dir,
        "--local_tmp_dir=%s" % tmp_dir,
        "--run_id=%s_%d" % (strategy, time.time()),
        "--epochs=%d" % epochs,
        "--generator_workers=40",
        "--sample_workers=8",
        "--batch_size=32",
        "--step_size_multiplier=1",
        "--neighborhood_workers=8",
        "--nocallbacks_copy_weights",
        "--num_gpus=1",
        #              "--autoencoder_loss_factor=.1",
        "--config=%s" % config
    ]
    app.run(main)
