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

"""This file contains experiments for using tf.similiar on fashion MNIST dataset.
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
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

flags.DEFINE_string("strategy", "stable_hard_quadruplet_loss",
                    "Moirai strategy to use.")

FLAGS = flags.FLAGS

# The directory we want to store our tensorboard visualzations.
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def read_fashion_mnist_data():
    """ Returns the fashion mnist data.

    Read the fashion mnist data from tf.keras.datasets and split
    the test dataset into test and target datasets.
    For more information on fashion mnist, please visit:
    https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles

    Returns:
        A tuple that contains three elements. The first element
        is a tuple that contains data used for training and
        the second element is a tuple that contains data used
        for testing. The third element is a tuple that contains
        the target data. All three tuples have the same
        structure, they contains two elements. The first
        element contains a dictionary for the specs of fashion mnist data
        (in 2d np array), the second element contains
        an np array of labels of class.
    """

    (x_train, y_train), (x_test_raw,
                         y_test_raw) = tf.keras.datasets.fashion_mnist.load_data()

    # Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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


def simple_fashion_mnist_tower_model():
    """A simple tower model for fashion mnist dataset.

    Returns:
        model: A tensorflow model that returns a 100-dimensional embedding.
    """

    i = Input(shape=(28, 28, 1), name="example")
    o = Conv2D(
        32,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1))(i)
    o = Conv2D(
        32,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1))(i)
    o = MaxPooling2D(pool_size=(2, 2))(o)
    o = Dropout(.25)(o)

    o = Conv2D(64, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(64, (3, 3), padding='same', activation='relu')(o)
    o = MaxPooling2D(pool_size=(2, 2))(o)
    o = Dropout(.25)(o)

    o = Flatten()(o)
    o = Dense(256, activation="relu")(o)
    o = Dropout(.25)(o)
    o = Dense(200)(o)
    model = Model(inputs=i, outputs=o)
    return model


class Normalize(Preprocessing):
    """A Preprocessing class that normalize the fashion MNIST example inputs."""

    def preprocess(self, img):
        """Normalize and reshape the input images."""

        normed = img["example"] / 255.0
        normed = normed.reshape((28, 28, 1))
        out = {"example": normed}
        return out


def display_metrics(test_metrics):
    unpacked_test_metrics = sorted([(i[0], i[1])
                                    for i in six.iteritems(test_metrics)])
    print("")
    print("TEST")
    print(tabulate.tabulate(unpacked_test_metrics, ["Metric", "Value"]))


def run_fashion_mnist_example(data, model, strategy, epochs):
    """An example usage of tf.similarity on fashion MNIST example.

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

    closest_items_plugin = ClosestItemsCallbackPlugin(
        closest_items_log_dir, image_key="example")

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
    data = read_fashion_mnist_data()
    model = simple_fashion_mnist_tower_model()

    strategy = FLAGS.strategy

    # This flag is inherited from tensorflow_similarity/common_flags.py
    epochs = FLAGS.epochs
    assert epochs > 0

    metrics = run_fashion_mnist_example(data, model, strategy, epochs)

    display_metrics(metrics)


if __name__ == '__main__':
    app.run(main)
