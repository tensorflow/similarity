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
import h5py
from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow_similarity.api.engine.augmentation import Augmentation
import numpy as np
import os
import six
import tabulate
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallback
from tensorflow_similarity.api.callbacks.plugins import ConfusionMatrixCallbackPlugin
from tensorflow_similarity.api.callbacks.plugins import EmbeddingProjectorCallbackPlugin
import datetime

DEFAULT_IRIS_DATA = os.path.join(os.path.dirname(__file__), "iris.csv")

flags.DEFINE_string("iris_csv", DEFAULT_IRIS_DATA, "Path to the iris dataset.")
flags.DEFINE_string("strategy", "stable_quadruplet_loss",
                    "SimHash strategy to use.")

FLAGS = flags.FLAGS

# The directory we want to store our tensorboard visualzations.
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def read_iris_data(data_path):
    """ Returns the iris data.

    Opens the data file specified by the argument, read each
    line and puts 20% of the data into the testing set.

    Args:
        data_path: A string that points to the iris dataset.

    Returns:
        A tuple that contains two elements. The first element
        is a tuple that contains data used for training and
        the second element is a tuple that contains data used
        for testing. Both of those two tuples have the same
        structure, they both contains two elements. The first
        element contains a dictionary for the specs of iris
        flowers (in 2d np array), the second element contains
        an np array of labels of class.
        For example:

        (
          ({'example': [[0,1,3,4],[2,1,3,5]]}, [0,2]),
          ({'example': [[0,2,3,5],[2,1,4,5]]}, [1,2])
        )
    """

    with tf.io.gfile.GFile(data_path, "r") as f:
        lines = f.readlines()
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for idx, line in enumerate(lines):
            tokens = line.split(",")
            y = int(tokens[-1])
            x = [float(i) for i in tokens[:-1]]

            if idx % 10 == 0:
                x_test.append(x)
                y_test.append(y)
            else:
                x_train.append(x)
                y_train.append(y)

        x_train = {"example": np.array(x_train)}
        x_test = {"example": np.array(x_test)}

        return ((x_train, np.array(y_train)), (x_test, np.array(y_test)))


def get_iris_data(data_path):
    """Computes and returns the training, testing, and target datasets."""

    (x_train, y_train), (x_test, y_test) = read_iris_data(data_path)
    (x_targets, y_targets) = create_targets(x_test, y_test)
    return (x_train, y_train), (x_test, y_test), (x_targets, y_targets)


def create_targets(x_test, y_test):
    """Creates targets from the test dataset.

    First we group the data by the labels (the value in y_test),
    then for each labels we compute the mean of the data.

    Args:
        x_test: A dictionary that contains a single key
            with the value of an 2d np array. For example,
            {"example": [[1,3,4,2], [2,1,4,7]]}
        y_test: A 1d np array containing the classification.
            For example: [0,1]

    Returns:
        x_targets: A dictionary that contains a single key
            with the value of an 2d np array. The length of
            the np array should be the number of classes.
        y_targets: A np.ndarry of shape(# examples, ) containing
            the labels for classification.
    """

    by_label = {0: [], 1: [], 2: []}

    for x, y in zip(x_test["example"], y_test):
        by_label[y].append(x)

    x_targets = []
    y_targets = []

    for label, data in six.iteritems(by_label):
        mean = np.mean(data, axis=0)
        x_targets.append(mean)
        y_targets.append(label)
    x_targets = np.array(x_targets)
    x_targets = {"example": x_targets}

    y_targets = np.array(y_targets)

    return x_targets, y_targets


def simple_model():
    """A simple tower model for iris dataset.

    Returns:
        model: A tensorflow model that returns a 3-dimensional embedding.
    """

    i = Input(shape=(4,), name='example')
    o = Dense(10, activation='tanh')(i)
    o = Dense(8, activation='tanh')(o)
    o = Dense(6, activation='tanh')(o)
    o = Dense(3)(o)
    model = Model(i, o)
    return model


def display_metrics(train_metrics, test_metrics):
    unpacked_train_metrics = [(i[0], i[1])
                              for i in six.iteritems(train_metrics)]
    unpacked_test_metrics = [(i[0], i[1]) for i in six.iteritems(test_metrics)]

    print("")
    print("TRAINING")
    print(tabulate.tabulate(unpacked_train_metrics, ["Metric", "Value"]))

    print("")
    print("TEST")
    print(tabulate.tabulate(unpacked_test_metrics, ["Metric", "Value"]))


class Fuzz(Augmentation):
    """An Augmentation class that disturbed the data."""

    def augment(self, x):
        """Returns disturbed data."""

        x = x["example"]

        FUZZ = .01
        fuzz = np.random.random_sample(x.shape) * FUZZ - FUZZ / 2.0

        x = x - fuzz
        return {"example": x}


def iris_nearest_neighbor_query(model, data):
    """Nearest neibhor query example."""

    _, _, (x_targets, y_targets) = data

    model.build_database(x_targets, y_targets)

    while True:
        print("Enter a iris's measurements to find its nearest neighbors, or 'quit' to end")

        petal_length = input("petal length? ")
        if petal_length == "quit":
            break
        petal_width = input("petal width? ")
        if petal_width == "quit":
            break
        sepal_length = input("sepal length? ")
        if sepal_length == "quit":
            break
        sepal_width = input("sepal width? ")
        if sepal_width == "quit":
            break

        example = {"example": np.array([
            [
                float(petal_length),
                float(petal_width),
                float(sepal_length),
                float(sepal_width)
            ]
        ])
        }

        neighbors = model.neighbors(example, N=3)
        print(neighbors)


def run_iris_example(data, strategy, tower_model, epochs):
    """A basic example usage of tf.similarity using iris dataset.

    This basic similarity run will first unpackage training,
    testing, and target data from the arguments and then construct a
    simple moirai model, fit the model with training data, then
    evaluate our model with training and testing datasets.

    Args:
        data: Sets, contains training, testing, and target datasets.
        strategy: String, specify the strategy to use for mining triplets.
        tower_model: tf.Model, the tower model to fit into moirai.
        epochs: Integer, number of epochs to fit our moirai model.
        callbacks: List of callback functions,

    Returns:
        moirai_model: SimHash
        train_metrics: Dictionary, containing metrics performed on the
            training dataset. The key is the name of the metric and the
            value is the np array of the metric values.
        test_metrics: Dictionary, containing metrics performed on the
            testing dataset. The key is the name of the metric and the
            value is the np array of the metric values.
    """

    # unpackage data
    (x_train, y_train), (x_test, y_test), (x_targets, y_targets) = data

    print("Initial tower model summary:")
    tower_model.summary()

    moirai_model = SimHash(
        tower_model,
        augmentation=Fuzz(),
        optimizer=Adam(lr=0.001),
        strategy=strategy)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", mode='min', min_delta=0.00000001, patience=50)

    confusion_matrix_log_dir = LOG_DIR + "/confusion_matrix"
    embedding_projector_log_dir = LOG_DIR + "/projector"

    confusion_matrix_plugin = ConfusionMatrixCallbackPlugin(
        confusion_matrix_log_dir)
    embedding_projector_plugin = EmbeddingProjectorCallbackPlugin(
        embedding_projector_log_dir)

    metrics_callbacks = MetricsCallback(
        [confusion_matrix_plugin, embedding_projector_plugin],
        x_test,
        y_test,
        x_targets,
        y_targets)

    moirai_model.fit(
        x_train,
        y_train,
        epochs=epochs,
        verbose=1,
        callbacks=[
            early_stopping_callback,
            metrics_callbacks
        ],
    )

    train_metrics = moirai_model.evaluate(
        x_train, y_train, x_targets, y_targets)
    test_metrics = moirai_model.evaluate(x_test, y_test, x_targets, y_targets)

    return moirai_model, train_metrics, test_metrics


def main(args):

    data = get_iris_data(FLAGS.iris_csv)
    tower_model = simple_model()
    strategy = FLAGS.strategy

    epochs = FLAGS.epochs
    assert epochs > 0

    model, train_metrics, test_metrics = run_iris_example(
        data, strategy, tower_model, epochs)

    display_metrics(train_metrics, test_metrics)
    iris_nearest_neighbor_query(model, data)


if __name__ == '__main__':
    app.run(main)
