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

import argparse
import datetime
import tempfile
from collections import defaultdict

import numpy as np
import six
import tabulate
import tensorflow as tf
from absl import app
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, GlobalMaxPool2D,
                                     Input, Lambda, MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_similarity.api.callbacks.tf_similarity_benchmark_callback import BenchmarkCallback
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow_datasets as tfds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy",
    type=str,
    default="triplet_loss",
    help="Moirai strategy to use.")
parser.add_argument(
    "--ways",
    type=int,
    default=100,
    help="The number of testing classes.")
parser.add_argument(
    "--shots",
    type=int,
    default=0,
    help="The number of target per testing class.")
parser.add_argument("--classes", type=int, default=500,
                    help="The number of training classes.")
parser.add_argument(
    "--samples",
    type=int,
    default=1000,
    help="The number of samples per training class.")
parser.add_argument(
    "--cnn_blocks",
    type=int,
    default=4,
    help="The number of cnn blocks for tower model.")
parser.add_argument(
    "--global_max_pool",
    type=bool,
    default=False,
    help="Whether to use globalMaxPool2D or gloablAvgPool2D layer.")
parser.add_argument("--regularize", type=bool, default=False,
                    help="Whether to regularize the embedding layer.")
parser.add_argument("--epochs", type=int, default=10,
                    help="The number of epochs to train the models.")

ARGS = parser.parse_args()


def read_omniglot_data(
        num_test_class=100,
        num_test_targets_per_class=0,
        num_train_class=500,
        num_train_samples_per_class=1000):
    """Returns the omniglot data.

    Read the omniglot data from tensorflow_datasets and split the test dataset
    into test and target datasets. For more information on omniglot dataset,
    please visit:
    https://github.com/brendenlake/omniglot
    https://www.tensorflow.org/datasets/catalog/omniglot

    Arguments:
        num_test_class (int): This is the number of classes we want to have
            in the test set. This parameter is equivant to the N in N-way,
            k-shots notation in single-shot learning papers. This will be
            capped to the maximum classes in the original test data set.
        num_test_targets_per_class (int): This is the number of targets we
            want to have per class in the test set. This parameter is
            equivant to the k in N-way, k-shots notation in single-shot learning
            papers. This will be capped to the minimum number of examples
            per test class in the test data set.
        num_train_class (int): This is the number of class we want to train on.
            This parameter will be capped at the maximum classes in the original
            train data set. Although single-shots papers currently do not
            cap the number of train classes/samples, it would be interesting to
            see how the number of training classes affect the generalization of
            our similarity models. This parameter is shorted handed to M-classes
            in the deisgn doc.
        num_train_samples_per_class (int): This is the number of samples per
            training class we want to train on. This parameter is shorted handed
            to l-samples in the design doc.

    Returns:
        A tuple that contains three elements. The first element
        is a tuple that contains data used for training and
        the second element is a tuple that contains data used
        for testing. The third element is a tuple that contains
        the target data. All three tuples have the same
        structure, they contains two elements. The first
        element contains a dictionary for the specs of omniglot data
        (in 2d np array), the second element contains
        an np array of labels of class.
    """

    # The classes in test_ds are already unseen in train_ds (the intersection
    # of classes in train_ds and the classes in test_ds is 0).
    train_ds, test_ds = tfds.load('omniglot:3.0.0', split=['train', 'test'])

    # fetch data from train_ds into arrays
    original_x_train = []
    original_y_train = []

    for sample in train_ds:
        label = sample["label"].numpy()
        image = sample["image"].numpy()[:, :, 0]
        original_x_train.append(image)
        original_y_train.append(label)

    # fetch data from test_ds into arrays
    original_x_test = []
    original_y_test = []

    for sample in test_ds:
        label = sample["label"].numpy()
        image = sample["image"].numpy()[:, :, 0]
        original_x_test.append(image)
        original_y_test.append(label)

    unique_y_test = np.unique(original_y_test)
    unique_y_train = np.unique(original_y_train)

    # cap the number of test and train class to the number of classes available
    num_test_class = min(num_test_class, len(unique_y_test))
    num_train_class = min(num_train_class, len(unique_y_train))

    # randomly select classes for N-way / M-classes for testing and training
    # respectively
    test_classes = set(
        np.random.choice(
            unique_y_test,
            num_test_class,
            replace=False))
    train_classes = set(
        np.random.choice(
            unique_y_train,
            num_train_class,
            replace=False))

    # build the training set
    x_train = []
    y_train = []
    train_seen = defaultdict(int)

    for image, label in zip(original_x_train, original_y_train):
        selected_class = label in train_classes
        not_sample_enough = train_seen[label] < num_train_samples_per_class
        if selected_class and not_sample_enough:
            train_seen[label] += 1
            x_train.append(image)
            y_train.append(label)

    # build the testing and target set
    x_tests = []
    y_tests = []
    x_targets = []
    y_targets = []

    target_seen = defaultdict(int)
    for image, label in zip(original_x_test, original_y_test):
        if label in test_classes:
            if target_seen[label] < num_test_targets_per_class:
                target_seen[label] += 1
                x_targets.append(image)
                y_targets.append(label)
            else:
                x_tests.append(image)
                y_tests.append(label)

    return (({
        "image": np.array(x_train)
    }, np.array(y_train)), ({
        "image": np.array(x_tests)
    }, np.array(y_tests)), ({
        "image": np.array(x_targets)
    }, np.array(y_targets)))


def simple_omniglot_tower_model(
        cnn_blocks=4,
        global_max_pooling=True,
        regularize=False):
    """A simple tower model for omniglot dataset.

    This model is the generalized version of VGG, users are able to specify the
    number of cnn blocks as well as the function for global pooling and whether
    or not to regularize the embedding layer.
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16
    The parameters of this model is setup such that users can experiment with
    different VGG-like architecture (as the baseline) to test the effectiveness
    of tf.similarity on popular single-shot learning benchmarks. Other
    state-of-the-art implmentations such as efficientNet would also be one
    of the default tower models that users can try, currently it is
    out-of-scope for this method.
    Efficient Net paper:
    https://arxiv.org/pdf/1905.11946.pdf

    Arguments:
        cnn_blocks (int): The number of CNN blocks. Every block consists of
            two Conv2D layers and a MaxPooling2D layer, the number of filters
            multiple by 2 for each additional block. See
            http://d2l.ai/chapter_convolutional-modern/vgg.html for more
            information about VGG block. Defaults to 4.
        global_max_pooling (boolean): If true, use GlobalMaxPool2D layer
            at the end of CNN blocks. Otherwise use GlobalAveragePooling2D
            layer. Defaults to True.
        regularize (boolean): If true, add a regularize layer at the end of
            the embedding layer. Defaults to False.

    Returns:
        model: A tensorflow model.
    """

    # define inputs
    i = Input(shape=(105, 105, 1), name="image")
    x = i

    # build CNN Blocks
    filters = 32
    for _ in range(cnn_blocks):
        x = Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu')(x)
        x = Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        filters *= 2

    # Select Global Pooling Layer
    if global_max_pooling:
        x = GlobalMaxPool2D()(x)
    else:
        x = GlobalAveragePooling2D()(x)

    # Build Fully Connected layers
    x = Dropout(.25)(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(32)(x)

    # Regularize embedding layer
    if regularize:
        x = Lambda(
            tf.math.l2_normalize,
            arguments={
                'axis': 1},
            name='normalize')(x)

    model = Model(inputs=i, outputs=x)
    return model


def vgg16_model_for_omniglot(dim=32):
    """A tower model that utilize the pretrained VGG16 on imageNet dataset.

    Returns:
        model: A tensorflow model that returns a 300-dimensional embedding.
    """

    i = Input(shape=(105, 105, 1), name="image")
    base_model = VGG16(input_tensor=i, weights=None, include_top=False)
    o = base_model.output

    # add a global spatial average pooling layer
    o = GlobalAveragePooling2D()(o)
    o = Dense(512, activation='relu')(o)
    o = Dense(256, activation="relu")(o)

    # add a r-dimension embedding layer
    o = Dense(dim)(o)

    model = Model(inputs=base_model.input, outputs=o)
    return model


class Normalize(Preprocessing):
    """A Preprocessing class that normalize the omniglot image inputs."""

    def preprocess(self, img):
        """Normalize and reshape the input images."""

        normed = img["image"] / 255.0
        normed = normed.reshape((105, 105, 1))
        out = {"image": normed}
        return out


def display_metrics(test_metrics):
    unpacked_test_metrics = sorted([(i[0], i[1])
                                    for i in six.iteritems(test_metrics)])
    print("")
    print("TEST")
    print(tabulate.tabulate(unpacked_test_metrics, ["Metric", "Value"]))


def run_omniglot_example(
        model=simple_omniglot_tower_model(),
        data=read_omniglot_data(),
        strategy="stable_quadruplet_loss",
        epochs="100",
        log_dir="logs",
        tensorboard_log_dir="tensorboard_logs/"):
    """An example usage of tf.similarity on omniglot example.

    This basic similarity run will first unpackage training,
    testing, and target data from the arguments and then construct a
    simple moirai model, fit the model with training data, then
    evaluate our model with training and testing datasets.

    Args:
        model: tf.Model, the tower model to fit into moirai.
        data: Sets, contains training, testing, and target datasets.
        strategy: String, specify the strategy to use for learning similarity.
        epochs: Integer, number of epochs to fit our moirai model.
        log_dir: String, the directory that we want to write benchmarks and
            tensorboard logs in.

    Returns:
        metrics: Dictionary, containing metrics performed on the
            testing dataset. The key is the name of the metric and the
            value is the np array of the metric values.
    """

    (x_train, y_train), (x_test, y_test), (x_targets, y_targets) = data

    moirai = SimHash(
        model,
        preprocessing=Normalize(),
        strategy=strategy,
        optimizer=Adam(lr=.001))

    benchmark_callback = BenchmarkCallback(
        x=x_test["image"],
        y=y_test,
        x_key="image",
        Ns=[5, 20, 50, 100, 200],
        ks=[1, 5, 10],
        log_dir=log_dir,
        tensorboard_log_dir=tensorboard_log_dir)

    callbacks = [benchmark_callback]

    moirai.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks)


def main(args):
    epochs = ARGS.epochs
    strategy = ARGS.strategy

    num_test_class = ARGS.ways
    num_test_targets_per_class = ARGS.shots
    num_train_class = ARGS.classes
    num_train_samples_per_class = ARGS.samples

    use_global_max_pooling = ARGS.global_max_pool
    regularize = ARGS.regularize
    cnn_blocks = ARGS.cnn_blocks

    data = read_omniglot_data(
        200,
        num_test_targets_per_class,
        num_train_class,
        num_train_samples_per_class)

    for strategy in [
        "triplet_loss",
        "quadruplet_loss",
            "stable_quadruplet_loss"]:
        for dim in [32, 64]:
            model = vgg16_model_for_omniglot(dim)
            log_dir = "logs/{}_{}_dimensions/".format(strategy, dim)
            tensorboard_log_dir = "tensorboard_logs/{}_{}_dimensions/".format(
                strategy, dim)

            run_omniglot_example(
                model,
                data,
                strategy,
                epochs,
                log_dir,
                tensorboard_log_dir)
    print("we are done!")


if __name__ == '__main__':
    app.run(main)
