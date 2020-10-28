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

"""This file contains experiments for using tf.similiar on CIFAR 10 dataset.
    Although GPU usage is not required, it is highly recommended.
"""

import datetime
import tempfile

import numpy as np
import six
import tabulate
import tensorflow as tf
from absl import app, flags
from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallback
from tensorflow_similarity.api.callbacks.plugins import (
    ClosestItemsCallbackPlugin, ConfusionMatrixCallbackPlugin)
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

flags.DEFINE_string("strategy", "stable_quadruplet_loss",
                    "Moirai strategy to use.")

FLAGS = flags.FLAGS

# The directory we want to store our tensorboard visualzations.
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def read_cifar10_data():
    """ Returns the CIFAR 10 data.

    Read the CIFAR 10 data from tf.keras.datasets and split
    the test dataset into test and target datasets.
    For more information on CIFAR 10, please visit:
    https://keras.io/datasets/#cifar10-small-image-classification

    Returns:
        A tuple that contains three elements. The first element
        is a tuple that contains data used for training and
        the second element is a tuple that contains data used
        for testing. The third element is a tuple that contains
        the target data. All three tuples have the same
        structure, they contains two elements. The first
        element contains a dictionary for the specs of CIFAR 10 data
        (in 2d np array), the second element contains
        an np array of labels of class.
    """

    (x_train, y_train), (x_test_raw,
                         y_test_raw) = tf.keras.datasets.cifar10.load_data()

    # flatten labels
    y_train = [label for [label] in y_train]
    y_test_raw = [label for [label] in y_test_raw]

    # Names of the integer classes, i.e., 0 -> airplane, 1 -> automobile, etc.
    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # re-label training and testing datasets
    y_train = [class_names[label] for label in y_train]
    y_test_raw = [class_names[label] for label in y_test_raw]

    x_tests = []
    y_tests = []

    x_targets = []
    y_targets = []

    seen = set()
    for x, y in zip(x_test_raw, y_test_raw):
        if y not in seen:
            seen.add(y)
            x_targets.append(x)
            y_targets.append(y)
        else:
            x_tests.append(x)
            y_tests.append(y)

    return (({
        "example": np.array(x_train)
    }, np.array(y_train)), ({
        "example": np.array(x_tests)
    }, np.array(y_tests)), ({
        "example": np.array(x_targets)
    }, np.array(y_targets)))


def pretrained_vgg16_model_for_cifar10():
    """A tower model that utilize the pretrained VGG16 on imageNet dataset.

    Returns:
        model: A tensorflow model that returns a 300-dimensional embedding.
    """

    i = Input(shape=(32, 32, 3), name="example")
    base_model = VGG16(input_tensor=i, weights='imagenet', include_top=False)
    o = base_model.output

    # add a global spatial average pooling layer
    o = GlobalAveragePooling2D()(o)
    o = Dense(1024, activation='relu')(o)

    # add a 300-dimension embedding layer
    o = Dense(300)(o)

    model = Model(inputs=base_model.input, outputs=o)
    return model


class Normalize(Preprocessing):
    """A Preprocessing class that normalize the CIFAR 10 example inputs."""

    def preprocess(self, img):
        """Normalize and reshape the input images."""

        normed = img["example"] / 255.0
        out = {"example": normed}
        return out


def display_metrics(test_metrics):
    unpacked_test_metrics = sorted([(i[0], i[1])
                                    for i in six.iteritems(test_metrics)])
    print("")
    print("TEST")
    print(tabulate.tabulate(unpacked_test_metrics, ["Metric", "Value"]))


def run_cifar10_example(data, model, strategy, epochs):
    """An example usage of tf.similarity on VGG 10 example.

    This basic similarity run will first unpackage training,
    testing, and target data from the arguments and then construct a
    simple moirai model, fit the model with training data, then
    evaluate our model with training and testing datasets.

    Args:
        data: Sets, contains training, testing, and target datasets.
        model: tf.Model, the tower model to fit into moirai.
        strategy: String, specify the strategy to use for learning similarity.
        epochs: Integer, number of epochs to fit our moirai model.

    Returns:
        metrics: Dictionary, containing metrics performed on the
            testing dataset. The key is the name of the metric and the
            value is the np array of the metric values.
    """

    # unpackage data
    (x_train, y_train), (x_test, y_test), (x_targets, y_targets) = data

    moirai = SimHash(
        model,
        preprocessing=Normalize(),
        strategy=strategy,
        optimizer=Adam(lr=.001),
        hard_mining_directory=tempfile.mkdtemp())

    loss_log_dir = LOG_DIR + "/loss"
    confusion_matrix_log_dir = LOG_DIR + "/confusion_matrix"
    closest_items_log_dir = LOG_DIR + "/closest_items"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=loss_log_dir)

    confusion_matrix_plugin = ConfusionMatrixCallbackPlugin(
        confusion_matrix_log_dir)

    closest_items_plugin = ClosestItemsCallbackPlugin(closest_items_log_dir)

    metrics_callbacks = MetricsCallback(
        [confusion_matrix_plugin, closest_items_plugin],
        x_test,
        y_test,
        x_targets,
        y_targets)

    callbacks = [tensorboard_callback, metrics_callbacks]

    moirai.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks)

    metrics = moirai.evaluate(x_test, y_test, x_targets, y_targets)
    return metrics


def main(args):
    data = read_cifar10_data()
    model = pretrained_vgg16_model_for_cifar10()

    strategy = FLAGS.strategy

    # This flag is inherited from tensorflow_similarity/common_flags.py
    epochs = FLAGS.epochs
    assert epochs > 0

    metrics = run_cifar10_example(data, model, strategy, epochs)

    display_metrics(metrics)


if __name__ == '__main__':
    app.run(main)
