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

from collections import defaultdict
import numpy as np
import six
import tensorflow as tf
from absl import app, flags
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPooling2D, Reshape, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tempfile
import tabulate
from tensorflow_similarity.api.engine.decoder import SimpleDecoder
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.simhash import SimHash
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask
from tensorflow_similarity.experiments.mnist.augments import ImageAugmentation

from tensorflow_similarity.api.callbacks.metrics_callbacks import MetricsCallback
from tensorflow_similarity.api.callbacks.plugins import EmbeddingProjectorCallbackPlugin
import datetime

flags.DEFINE_string(
    "data_path", "./mnist.npz",
    "Path to where the cached MNIST data is/should be stored.")

flags.DEFINE_string("strategy", "stable_quadruplet_loss",
                    "Moirai strategy to use.")

flags.DEFINE_boolean("augment", False,
                     "If true, augment data.")

flags.DEFINE_boolean("autoencoder", False,
                     "If true, use an auxillary autoencoder task.")

flags.DEFINE_boolean(
    "few_shot_learning",
    False,
    "If true, we will filter the training dataset to do few-shot learning.")

flags.DEFINE_integer(
    "few_shot_samples",
    10,
    "This integer sets the number of examples we will train on for the filtered class. Only useful when few_shot_learning is True. Defaults to 10."
)

flags.DEFINE_boolean(
    "train_even",
    True,
    "If true, we will train on examples from even number classes and filtered out the odd number classes for few-shot learning, vice versa for False. Only useful when few_shot_learning is True."
)

FLAGS = flags.FLAGS

# The directory we want to store our tensorboard visualzations.
LOG_DIR = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def few_shot_preprocess(x_train, y_train, n, train_even):
    """ Returns training data for few/zero shot training.

    Arguments:
        x_train (np.array): An array that holds the training data, where
            x_train[i] holds the i'th example.
        y_train (np.array): An array that holds the training labels, where
            y_train[j] holds the label for x_train[j].
        n (int): Number of examples we want to train on for the filtered class.
            If n == 0 then we have zero-shot learning (training on even
                number classes only (digits that are even in MNIST dataset)
                if train_even is True, otherwise training on odd
                number classes).
            If n == 1 it is one-shot learning (get 2 examples for each odd
                number because we need 2 example for one-shot learning for
                similiarity learning).
            If n > 1 it is few shots learning.
            If n >= number of training samples then this method will return
                the original x_train and y_train provided.
        train_even (boolean): Whether we train on evens or odds

    Returns:
        filtered_x_train (np.array): An array that holds the filtered
            training data, where filtered_x_train[i] holds the i'th example.
        filtered_y_train (np.array): An array that holds the filtered
            training labels, where filtered_y_train[j] holds the label
            for filtered_x_train[j].
    """

    if n >= len(x_train):
        return x_train, y_train

    # in triplet/quadruplet loss learning we need 2 example to be considered
    # one shot learning.
    if n == 1:
        n = 2

    seen = defaultdict(int)
    filtered_x_train = []
    filtered_y_train = []
    for x, y in zip(x_train, y_train):
        is_even = (y % 2 == 0)
        not_seen_enough = seen[y] < n
        if is_even == train_even or not_seen_enough:
            seen[y] += 1
            filtered_x_train.append(x)
            filtered_y_train.append(y)

    return filtered_x_train, filtered_y_train


def read_mnist_data(data_path):
    """ Returns the mnist data.

    Read the mnist data from tf.keras.datasets and split the test dataset
    into test and target datasets.
    For more information on mnist, please visit:
    https://keras.io/datasets/#mnist-database-of-handwritten-digits

    Args:
        data_path: A string that points to the cached mnist
            dataset.

    Returns:
        A tuple that contains three elements. The first element
        is a tuple that contains data used for training and
        the second element is a tuple that contains data used
        for testing. The third element is a tuple that contains
        the target data. All three tuples have the same
        structure, they contains two elements. The first
        element contains a dictionary for the specs of mnist data
        (in 2d np array), the second element contains
        an np array of labels of class.
    """

    (x_train, y_train), (x_test_raw,
                         y_test_raw) = tf.keras.datasets.mnist.load_data(path=data_path)

    if FLAGS.few_shot_learning:
        x_train, y_train = few_shot_preprocess(
            x_train, y_train, FLAGS.few_shot_samples, FLAGS.train_even)

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


def simple_mnist_tower_model():
    """A simple tower model for mnist dataset.

    Returns:
        model: A tensorflow model.
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
    o = Dense(100)(o)
    model = Model(inputs=i, outputs=o)
    return model


class MNISTDecoder(SimpleDecoder):
    """A Decoder class for MNIST dataset."""

    def build_reconstruction_model(self):
        """A model that reconstruct MNIST features from embedding."""

        i = self.create_embedding_input()

        x = Dense(128)(i)
        x = Reshape((4, 4, 8), name="reshape_input")(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        x = Flatten()(x)
        x = self.feature_shaped_dense(x)

        m = Model(inputs=i, outputs=[x])
        return m


class Normalize(Preprocessing):
    """A Preprocessing class that normalize the MNIST example inputs."""

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


def run_mnist_example(
        data,
        model,
        strategy,
        augment,
        autoencoder,
        epochs,
        prewarm_epochs):
    """An example usage of tf.similarity on MNIST example.

    This basic similarity run will first unpackage training,
    testing, and target data from the arguments and then construct a
    simple moirai model, fit the model with training data, then
    evaluate our model with training and testing datasets.

    Args:
        data: Sets, contains training, testing, and target datasets.
        model: tf.Model, the tower model to fit into moirai.
        strategy: String, specify the strategy to use for learning similarity.
        augment: Boolean, indicates whether we want to augment our data.
        epochs: Integer, number of epochs to fit our moirai model.
        prewarm_epochs: Integer, number of prewarm epochs to fit our moirai model.

    Returns:
        metrics: Dictionary, containing metrics performed on the
            testing dataset. The key is the name of the metric and the
            value is the np array of the metric values.
    """

    # unpackage data
    (x_train, y_train), (x_test, y_test), (x_targets, y_targets) = data

    aux_tasks = []
    if autoencoder:
        task = AutoencoderTask("ae",
                               model,
                               MNISTDecoder(name="mnist_decode"),
                               tower_names=["anchor"],
                               field_names=["example"],
                               **kwargs)
        aux_tasks.append(task)

    aug = None
    if augment:
        aug = ImageAugmentation()

    moirai = SimHash(
        model,
        auxillary_tasks=aux_tasks,
        augmentation=aug,
        preprocessing=Normalize(),
        strategy=strategy,
        optimizer=Adam(lr=.001),
        hard_mining_directory=tempfile.mkdtemp())

    loss_log_dir = LOG_DIR + "/loss"
    embedding_projector_log_dir = LOG_DIR + "/projector"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=loss_log_dir)

    embedding_projector_plugin = EmbeddingProjectorCallbackPlugin(
        embedding_projector_log_dir, sprite_image_key="example")

    metrics_callbacks = MetricsCallback(
        [embedding_projector_plugin],
        x_test,
        y_test,
        x_targets,
        y_targets)

    callbacks = [tensorboard_callback, metrics_callbacks]

    moirai.fit(
        x_train,
        y_train,
        prewarm_epochs=prewarm_epochs if autoencoder else 0,
        epochs=epochs,
        callbacks=callbacks)

    metrics = moirai.evaluate(x_test, y_test, x_targets, y_targets)
    return metrics


def main(args):
    data = read_mnist_data(FLAGS.data_path)
    model = simple_mnist_tower_model()

    strategy = FLAGS.strategy
    augment = FLAGS.augment
    autoencoder = FLAGS.autoencoder

    # those two flags are inherited from tensorflow_similarity/common_flags.py
    epochs = FLAGS.epochs
    prewarm_epochs = FLAGS.prewarm_epochs

    assert epochs > 0
    assert prewarm_epochs >= 0

    metrics = run_mnist_example(
        data,
        model,
        strategy,
        augment,
        autoencoder,
        epochs,
        prewarm_epochs)

    display_metrics(metrics)


if __name__ == '__main__':
    app.run(main)
