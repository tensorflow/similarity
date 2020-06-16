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
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from miniimagenettools.mini_imagenet_dataloader import MiniImageNetDataLoader
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

import tensorflow_datasets as tfds

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10,
                    help="The number of epochs to train the models.")

ARGS = parser.parse_args()


def read_omniglot_data(
        num_test_class=100,
        num_test_targets_per_class=0,
        num_train_class=500,
        num_train_samples_per_class=20):
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


def model_builder(dim=32, model="VGG16", is_omniglot=True):
    """A tower model that utilize the pretrained VGG16 on imageNet dataset.

    Returns:
        model: A tensorflow model that returns a dim-dimensional embedding.
    """

    if is_omniglot:
        i = Input(shape=(105, 105, 1), name="image")
    else:
        # miniImageNet data
        i = Input(shape=(84, 84, 3), name="image")
    if model == "VGG16":
        base_model = VGG16(
            input_tensor=i,
            weights=None,
            include_top=False,
            pooling=None)
    elif model == "DenseNet":
        base_model = DenseNet121(
            input_tensor=i,
            weights=None,
            include_top=False)
    elif model == "ResNet":
        base_model = ResNet50V2(
            input_tensor=i,
            weights=None,
            include_top=False)
    o = base_model.output

    if is_omniglot:
        # o = GlobalAveragePooling2D()(o)
        o = Flatten()(o)
        o = Dense(512, activation='relu')(o)
        o = Dense(256, activation="relu")(o)
    else:
        # o = GlobalMaxPool2D()(o)
        o = Flatten()(o)
        o = Dense(512 * 2, activation='relu')(o)
        o = Dense(256 * 2, activation="relu")(o)

    # add a r-dimension embedding layer
    o = Dense(dim)(o)

    model = Model(inputs=base_model.input, outputs=o)
    return model


class Normalize(Preprocessing):
    """A Preprocessing class that normalize the omniglot image inputs."""

    def __init__(self, image_shape):
        self.image_shape = image_shape

        super(Normalize, self).__init__()

    def preprocess(self, img):
        """Normalize and reshape the input images."""

        normed = img["image"] / 255.0
        normed = normed.reshape(self.image_shape)
        out = {"image": normed}
        return out


def display_metrics(test_metrics):
    unpacked_test_metrics = sorted([(i[0], i[1])
                                    for i in six.iteritems(test_metrics)])
    print("")
    print("TEST")
    print(tabulate.tabulate(unpacked_test_metrics, ["Metric", "Value"]))


def run_benchmark_example(
        model=None,
        data=None,
        strategy="stable_quadruplet_loss",
        epochs="100",
        log_dir="logs",
        tensorboard_log_dir="tensorboard_logs/",
        is_mini_imagenet=False):
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

    shape = (105, 105, 1)
    if is_mini_imagenet:
        shape = (84, 84, 3)

    preprocessing = Normalize(shape)
    if is_mini_imagenet:
        preprocessing = None

    moirai = SimHash(
        model,
        preprocessing=preprocessing,
        strategy=strategy,
        optimizer=Adam(lr=.001))

    Ns = [5, 20, 50, 100, 200]
    ks = [1, 5, 10]
    if is_mini_imagenet:
        Ns = [5, 20]

    benchmark_callback = BenchmarkCallback(
        x=x_test["image"],
        y=y_test,
        x_key="image",
        Ns=Ns,
        ks=ks,
        log_dir=log_dir,
        tensorboard_log_dir=tensorboard_log_dir)

    callbacks = [benchmark_callback]

    moirai.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=callbacks)


def read_mini_imagenet_data(
        num_train_class=10,
        num_train_samples_per_class=100):
    # we only need train and test datasets
    train_data_loader = MiniImageNetDataLoader(
        shot_num=num_train_samples_per_class - 1,
        way_num=num_train_class,
        episode_test_sample_num=1)

    train_data_loader.generate_data_list(phase='train', episode_num=1)
    train_data_loader.load_list(phase='train')

    episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
        train_data_loader.get_batch(phase='train', idx=0)

    training_data = np.array(episode_train_img)

    x_train = {"image": training_data}

    episode_train_label = np.array(
        [np.where(r == 1)[0][0] for r in episode_train_label])

    training_label = np.array(episode_train_label)

    y_train = np.array(training_label)

    # get test data
    test_data_loader = MiniImageNetDataLoader(
        shot_num=600 - 1,
        way_num=20,
        episode_test_sample_num=1)

    test_data_loader.generate_data_list(phase='test', episode_num=1)
    test_data_loader.load_list(phase='test')

    episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
        test_data_loader.get_batch(phase='test', idx=0)

    test_data = np.array(episode_train_img)

    x_test = {"image": test_data}

    episode_train_label = np.array(
        [np.where(r == 1)[0][0] for r in episode_train_label])

    test_label = np.array(episode_train_label)

    y_test = np.array(test_label)

    return (x_train, y_train), (x_test, y_test), (None, None)


def main(args):
    # benchmark on omniglot and
    epochs = ARGS.epochs

    # get mini_imageNet data
    mini_imagenet_data_storage = dict()

    small_mini_imagenet_data = read_mini_imagenet_data(
        num_train_class=20, num_train_samples_per_class=50)

    larget_mini_imagenet_data = read_mini_imagenet_data(
        num_train_class=64, num_train_samples_per_class=600)

    mini_imagenet_data_storage["small"] = small_mini_imagenet_data
    mini_imagenet_data_storage["large"] = larget_mini_imagenet_data

    # get omniglot data
    omniglot_data_storage = dict()
    small_omniglot_training_data = read_omniglot_data(
        200,
        0,
        50,
        20)
    omniglot_data_storage["small"] = small_omniglot_training_data
    large_omniglot_training_data = read_omniglot_data(
        200,
        0,
        1000,
        20)
    omniglot_data_storage["large"] = large_omniglot_training_data

    strategies = [
        "quadruplet_loss",
        "triplet_loss",
        "stable_quadruplet_loss"]
    dimensions = [32, 64]
    data_sizes = ["large", "small"]
    model_architectures = ["VGG16"]

    for strategy in strategies:
        for dim in dimensions:
            for data_size in data_sizes:
                for architecture in model_architectures:

                    print(
                        "start to train on mini imagenet with {} architecture, {} strategy, {} dimensions, {} training size.".format(
                            architecture,
                            strategy,
                            dim,
                            data_size))
                    # benchmark for miniImageNet
                    mini_imagenet_data = mini_imagenet_data_storage[data_size]
                    mini_imagenet_model = model_builder(
                        dim, architecture, False)
                    mini_imagenet_log_dir = "mini_imagenet_logs/{}_{}_{}_dim_{}_train_size/".format(
                        architecture, strategy, dim, data_size)
                    mini_imagenet_tensorboard_log_dir = "mini_imagenet_tensorboard_logs/{}_{}_{}_dim_{}_train_size/".format(
                        architecture, strategy, dim, data_size)

                    run_benchmark_example(
                        mini_imagenet_model,
                        mini_imagenet_data,
                        strategy,
                        epochs,
                        mini_imagenet_log_dir,
                        mini_imagenet_tensorboard_log_dir,
                        is_mini_imagenet=True)

                    print(
                        "start to train on omniglot with {} architecture, {} strategy, {} dimensions, {} training size.".format(
                            architecture, strategy, dim, data_size))
                    # benchmark for omniglot
                    omniglot_data = omniglot_data_storage[data_size]
                    omniglot_model = model_builder(
                        dim, architecture, is_omniglot=True)
                    omniglot_log_dir = "omniglot_logs/{}_{}_{}_dim_{}_train_size/".format(
                        architecture, strategy, dim, data_size)
                    omniglot_tensorboard_log_dir = "omniglot_tensorboard_logs/{}_{}_{}_dim_{}_train_size/".format(
                        architecture, strategy, dim, data_size)

                    run_benchmark_example(
                        omniglot_model,
                        omniglot_data,
                        strategy,
                        epochs,
                        omniglot_log_dir,
                        omniglot_tensorboard_log_dir)

    print("we are done!")


if __name__ == '__main__':
    app.run(main)
