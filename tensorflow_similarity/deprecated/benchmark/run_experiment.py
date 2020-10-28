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

"""This file contains experiments for using tf.similiar on omniglot dataset.
    Although GPU usage is not required, it is highly recommended.
"""

from collections import defaultdict

import numpy as np
import tensorflow as tf
from miniimagenettools.mini_imagenet_dataloader import MiniImageNetDataLoader
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow_similarity.api.callbacks.benchmark import BenchmarkCallback
from tensorflow_similarity.api.engine.simhash import SimHash

import tensorflow_datasets as tfds

from .data_loader import get_dataset


def run_experiment(config):
    """Run benchmark experiment given the parameters specified in the config
       file.

    Args:
        config (Dict): A dictionary contains parameters for the experiment.

    Returns:
        output_path (String): The path to the output file that stores the
            benchmark data.
    """

    #import ipdb
    # ipdb.set_trace()
    # experiment: limit GPU memory growth to see if it will stop big models
    # from crashing.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tf.keras.backend.clear_session()

    # model information
    tower_model_architecture = config.get("tower_model_architecture", "VGG16")
    embedding_size = config.get("embedding_size", 32)
    similarity_loss = config.get("similarity_loss", "triplet_loss")
    auxiliary_task = config.get("auxiliary_task", None)

    # package model information into a dictionary to pass it to Benchmark
    model_information = dict()
    model_information["tower_model_architecture"] = tower_model_architecture
    model_information["embedding_size"] = embedding_size
    model_information["similarity_loss"] = similarity_loss
    model_information["auxiliary_task"] = auxiliary_task

    # training information
    num_epochs = config.get("num_epochs", 100)

    # data information
    Ns = config.get("num_unseen_classes", [5])
    ks = config.get("num_targets", [1])
    percentage_training_class = config.get("training_classes_percentage", 1.0)
    num_training_samples = config.get("num_training_samples", 100)
    dataset = config.get("dataset", "omniglot")

    # sort Ns and ks
    Ns.sort()
    ks.sort()

    # package model information into a dictionary to pass it to Benchmark
    data_information = dict()
    data_information["num_unseen_classes"] = Ns
    data_information["num_targets"] = ks
    data_information["training_classes_percentage"] = percentage_training_class
    data_information["num_training_samples"] = num_training_samples
    data_information["dataset"] = dataset

    output_path = config.get("output_path", "/benchmark")
    output_file_name = config.get("output_file_name", None)
    # generate output file name if user did not specify one
    if not output_file_name:
        output_file_name = "{}_{}dim_{}_{}_{}_classes_{}_samples.json".format(
            tower_model_architecture,
            embedding_size,
            similarity_loss,
            dataset,
            int(percentage_training_class * 100),
            num_training_samples)

    # fetch the data
    largest_ways = Ns[-1]
    (x_train, y_train), (x_test, y_test) = get_dataset(
        dataset, largest_ways, percentage_training_class, num_training_samples)

    num_train_classes = len(np.unique(y_train))
    data_information["num_seen_classes"] = num_train_classes

    # create tower model based on model information
    input_shape = x_train["image"][0].shape

    tower_model = get_tower_model(
        input_shape,
        tower_model_architecture,
        embedding_size)

    model_information["tower_model_params"] = tower_model.count_params()
    model_information["tower_model_layers"] = len(tower_model.layers)

    '''
    # this block of code test if regular ResNet50 + Beam works
    # we conclude it works
    # remove the last (prediction) layer, we will swap this layer with
    # the embedding layer
    base_model = Model(tower_model.input, tower_model.layers[-2].output)
    o = base_model.output
    o = Dense(len(np.unique(y_train)))(o)
    model = Model(inputs=base_model.input, outputs=o)

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=256 * 4, epochs=1)

    '''
    # create the Tensorflow Similarity model
    similarity_model = SimHash(
        tower_model,
        batch_size=32,
        strategy=similarity_loss)

    # create the benchmark callback
    benchmark_callback = BenchmarkCallback(
        x=x_test["image"],
        y=y_test,
        x_key="image",
        Ns=Ns,
        ks=ks,
        model_information=model_information,
        data_information=data_information,
        log_dir=output_path,
        file_name=output_file_name)
    callbacks = [benchmark_callback]

    # run the model
    similarity_model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        generator_workers=1,
        callbacks=callbacks)

    # return the path to benchmark data file
    return '{}/{}'.format(output_path, output_file_name)


def get_tower_model(
        input_shape,
        tower_model_architecture="VGG16",
        embedding_size=32):
    """Returns tower model with the given architecture and embedding size.

    Args:
        input_shape (tuple): Tuple of shape for the input to the twoer model.
        tower_model_architecture (str, optional): The architecture, such as
            VGG, ResNet, that the user wants to deploy. Defaults to "VGG16".
        embedding_size (int, optional): The size of embedding (output) layer.
            Defaults to 32.

    Returns:
        model: A tensorflow model that returns a dim-dimensional embedding.
    """

    i = Input(shape=input_shape, name="image")

    # currently we only support VGG and ResNet
    if tower_model_architecture == "VGG16":
        base_model = VGG16(
            input_tensor=i,
            weights=None,
            include_top=True)
    elif tower_model_architecture == "ResNet":
        base_model = ResNet50V2(
            input_tensor=i,
            weights=None,
            include_top=True)

    # remove the last (prediction) layer, we will swap this layer with
    # the embedding layer
    base_model = Model(base_model.input, base_model.layers[-2].output)
    o = base_model.output

    # add a embedding layer sepcified by user
    o = Dense(embedding_size, name="Embedding")(o)
    o = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2_norm")(o)

    model = Model(inputs=base_model.input, outputs=o)
    return model
