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
import tensorflow as tf
from miniimagenettools.mini_imagenet_dataloader import MiniImageNetDataLoader
from collections import defaultdict

import tensorflow_datasets as tfds


def get_dataset(
        dataset="omniglot",
        num_unseen_classes=5,
        percentage_train_class=1.0,
        num_train_samples=100):
    """Return the data for training and testing/benchmarking.

    Args:
        dataset (str, optional): The dataset we want to benchmark
            currently only support omniglot and miniImageNet. Defaults to "omniglot".
        num_unseen_classes (int, optional): The number of unseen class that we
            want to include in the testing dataset. Defaults to 5.
        percentage_train_class (float, optional): The percentage of training class
            we want to include in the training class. This helps users
            to benchmark how many percentage of training dataset is needed to
            achieve good result. Defaults to 1.0.
        num_train_samples (int, optional): [description]. Defaults to 100.
    """

    result = None

    if dataset == "omniglot":
        result = get_omniglot_data(
            num_unseen_classes,
            percentage_train_class,
            num_train_samples)
    elif dataset == "mini_imagenet":
        result = get_mini_imagenet_data(
            num_unseen_classes,
            percentage_train_class,
            num_train_samples)
    elif dataset == "cifar100":
        result = get_cifar100_data(
            num_unseen_classes,
            percentage_train_class,
            num_train_samples)

    return result


def get_cifar100_data(
        num_unseen_classes=20,
        percentage_train_class=1.0,
        num_train_samples=20):

    # get the data from tf.keras datasets
    # we want to split this train and test dataset differently than what
    # is in the original dataset split, we want the data in test dataset
    # to be unseen in the training dataset.
    (x_train_raw, y_train_raw), (x_test_raw,
                                 y_test_raw) = tf.keras.datasets.cifar100.load_data()

    # gather all the data
    x = np.concatenate((x_train_raw, x_test_raw))
    y = np.concatenate((y_train_raw, y_test_raw))
    y = y.flatten()

    unique_classes = np.unique(y)

    # sample n classes for testing
    test_classes = np.random.choice(
        unique_classes,
        num_unseen_classes,
        replace=False)

    # classes that are not selected are available for training
    available_train_classes = np.setdiff1d(
        unique_classes, test_classes, assume_unique=True)

    num_train_classes = int(
        len(available_train_classes * percentage_train_class))

    train_classes = np.random.choice(
        available_train_classes,
        num_train_classes,
        replace=False)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_seen = defaultdict(int)

    # iterate through the cifar100 dataset and partition the data for
    # few shots learning
    for image, label in zip(x, y):
        if label in test_classes:
            x_test.append(image)
            y_test.append(label)
        elif label in train_classes and train_seen[label] < num_train_samples:
            train_seen[label] += 1
            x_train.append(image)
            y_train.append(label)

    x_train = {"image": np.array(x_train)}
    y_train = np.array(y_train)
    x_test = {"image": np.array(x_test)}
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def get_omniglot_data(
        num_unseen_classes=20,
        percentage_train_class=1.0,
        num_train_samples=20):

    # The classes in test_ds are already unseen in train_ds (the intersection
    # of classes in train_ds and the classes in test_ds is 0).
    train_ds, test_ds = tfds.load('omniglot:3.0.0', split=['train', 'test'])

    # fetch data from train_ds into arrays
    original_x_train = []
    original_y_train = []

    for sample in train_ds:
        label = sample["label"].numpy()
        image = sample["image"].numpy() / 255
        original_x_train.append(image)
        original_y_train.append(label)

    # fetch data from test_ds into arrays
    original_x_test = []
    original_y_test = []

    for sample in test_ds:
        label = sample["label"].numpy()
        image = sample["image"].numpy() / 255
        original_x_test.append(image)
        original_y_test.append(label)

    unique_y_test = np.unique(original_y_test)
    unique_y_train = np.unique(original_y_train)

    # compute the number of training_class
    num_train_class = int(len(unique_y_train) * percentage_train_class)

    # cap the number of test class to the number of classes available
    num_test_class = min(num_unseen_classes, len(unique_y_test))

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
        not_sample_enough = train_seen[label] < num_train_samples
        if selected_class and not_sample_enough:
            train_seen[label] += 1
            x_train.append(image)
            y_train.append(label)

    # build the test dataset
    x_test = []
    y_test = []

    for image, label in zip(original_x_test, original_y_test):
        if label in test_classes:
            x_test.append(image)
            y_test.append(label)

    x_train = {"image": np.array(x_train)}
    y_train = np.array(y_train)
    x_test = {"image": np.array(x_test)}
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def get_mini_imagenet_data(
        num_unseen_classes,
        percentage_train_class,
        num_train_samples):

    # TODO(shunlin): currently we are relying on mini-imagenet-tools to
    # get the data, in order to use this the user would need to pip install
    # mini-imagenet-tool and also manually download the data from the Github
    # page and put the processed data in the current folder.
    # we would want to use tfds if possible but it's currently still been
    # developed.

    # The total number of available class for training in the miniImageNet is
    # 64, see https://github.com/yaoyao-liu/mini-imagenet-tools for details
    num_available_training_class = 64
    num_train_class = int(
        percentage_train_class *
        num_available_training_class)

    # due to how the miniImageNet tool is setup, we need at least 1 sample
    # for the "test" dataset, thus we cap it at 599.
    num_train_samples = min(599, num_train_samples)
    train_data_loader = MiniImageNetDataLoader(
        shot_num=num_train_samples,
        way_num=num_train_class,
        episode_test_sample_num=1)

    train_data_loader.generate_data_list(phase='train', episode_num=1)
    train_data_loader.load_list(phase='train')

    episode_train_img, episode_train_label, _, _ = \
        train_data_loader.get_batch(phase='train', idx=0)

    training_data = np.array(episode_train_img)

    x_train = {"image": training_data}

    episode_train_label = np.array(
        [np.where(r == 1)[0][0] for r in episode_train_label])

    training_label = np.array(episode_train_label)

    y_train = np.array(training_label)

    # cap the number of unseen classes at 20 because that is the maximum
    # that is allowed for mini-ImageNet tool
    num_unseen_classes = min(20, num_unseen_classes)

    # get test data
    test_data_loader = MiniImageNetDataLoader(
        shot_num=600 - 1,
        way_num=20,
        episode_test_sample_num=1)

    test_data_loader.generate_data_list(phase='test', episode_num=1)
    test_data_loader.load_list(phase='test')

    episode_train_img, episode_train_label, _, _ = \
        test_data_loader.get_batch(phase='test', idx=0)

    test_data = np.array(episode_train_img)

    x_test = {"image": test_data}

    episode_train_label = np.array(
        [np.where(r == 1)[0][0] for r in episode_train_label])

    test_label = np.array(episode_train_label)

    y_test = np.array(test_label)

    return (x_train, y_train), (x_test, y_test)
