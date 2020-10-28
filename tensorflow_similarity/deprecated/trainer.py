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
import collections
import importlib
import json
import tensorflow_similarity
from tensorflow_similarity import *
import tensorflow_similarity.architectures
import tensorflow_similarity.losses
import tensorflow_similarity.generators
import tensorflow_similarity.strategies
import tensorflow_similarity.readers
import tensorflow_similarity.callbacks
import tensorflow_similarity.preprocessing
from tensorflow_similarity.utils.config_utils import load_custom_configuration
from tensorflow_similarity.architectures.architectures import *
from tensorflow_similarity.recipes.recipe_registry import get_recipe
from tensorflow_similarity.architectures.model_registry import get_model
from tensorflow_similarity.generators.base import MoiraiGenerator
from tensorflow_similarity.model import Moirai
import sys
import tensorflow as tf
from tensorflow.keras.backend import set_session
import time
import traceback
import os
import warnings
from tensorflow.keras.optimizers import deserialize as deserialize_optimizer, serialize as serialize_optimizer, Optimizer, Adam

flags.DEFINE_boolean("log_device_placement", False, "")

FLAGS = flags.FLAGS


def train(reader,
          keras_checkpoint=None,
          dataset=None,
          augmentation=None,
          augmented_dataset=None,
          filter=None,
          validation_reader=None,
          validation_dataset=None,
          epochs=100,
          prewarm_epochs=10,
          model_name=None,
          model=None,
          decoder_model_name=None,
          decoder_model=None,
          decoder_losss=None,
          strategy_name=None,
          strategy=None,
          num_gpus=None,
          visualizations=[],
          optimizer=Adam(lr=.001),
          callbacks=[]):

    if num_gpus:
        FLAGS.batch_size *= num_gpus

    # Configure Keras to A) only use as much GPU memory as necessary,
    # and B) optionally log where operations occur
    config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement,
        allow_soft_placement=True)
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(
        sess)  # set this TensorFlow session as the default session for Keras

    load_custom_configuration()

    if not strategy:
        strategy = get_recipe(strategy_name)

    if not model:
        model = get_model(model_name)

    if not decoder_model:
        if decoder_model_name:
            decoder_model = get_model(decoder_model_name)

    moirai_model = strategy(
        model,
        decoder_model=decoder_model,
        reader=reader,
        dataset_config=dataset,
        filter=filter,
        augmentation=augmentation,
        augmented_dataset_config=augmented_dataset,
        validation_reader=validation_reader,
        validation_dataset=validation_dataset,
        extra_callbacks=callbacks,
        num_gpus=num_gpus,
        decoder_losss=decoder_losss,
        visualizations=visualizations,
        optimizer=optimizer,
        is_sampled=False)

    moirai_model.write_config()

    if keras_checkpoint:
        try:
            print("Loading model from: %s" % keras_checkpoint)
            sess = tf.Session()
            with sess.as_default():
                prior_model = tf.keras.models.load_model(keras_checkpoint)
                w = prior_model.get_weights()
                sess = None
            moirai_model.get_inference_model().set_weights(w)
        except BaseException:
            traceback.print_exc()
            print("=============================================")
            print("COULD NOT LOAD CHECKPOINT. Continue? (Y/n)")
            print("=============================================")
            r = input("> ")

            if r[0] != "Y" and r[0] != "y":
                exit()

    moirai_model.train(epochs=epochs, prewarm_epochs=prewarm_epochs)
