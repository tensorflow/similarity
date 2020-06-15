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

import collections
import copy
import random
import sys
import traceback

import numpy as np
import numpy.random as rng
import six
from tqdm import tqdm

from tensorflow_similarity.utils.config_utils import (
    deserialize_moirai_object, json_dict_to_moirai_obj, register_custom_object,
    serialize_moirai_object, value_or_callable)


class FeatureSpec(object):
    def __init__(self,
                 name,
                 feature_name=None,
                 preprocessing=None,
                 optional=False):
        self.name = name
        if feature_name:
            self.feature_name = feature_name
        else:
            self.feature_name = name

        self.preprocessing = deserialize_moirai_object(preprocessing)
        self.optional = optional

    def get_config(self):
        return {
            "name":
            self.name,
            "feature_name":
            self.feature_name,
            "preprocessing":
            self.preprocessing.get_config()
            if self.preprocessing is not None else None,
            "optional":
            self.optional
        }


register_custom_object("FeatureSpec", FeatureSpec)


def deserialize_featurespec(v):
    if not v:
        return None
    feature = deserialize_moirai_object(v)
    if isinstance(feature, six.string_types):
        feature = FeatureSpec(feature, feature_name=feature)
    return feature


Datum = collections.namedtuple("Datum", [
    'index', 'id', 'feature_dictionary', 'raw_feature_dictionary', 'label',
    'metadata', 'group'
])


class Dataset(object):
    def __init__(self, data, dataset_config):
        """
        data: dictionary of feature name to an array-like containing the features.

        """
        self.data = data
        self.dataset_config = dataset_config
        self.class_index = []
        self.length = len(data[list(data.keys())[0]])
        self.num_classes = 0

        labels = None
        if self.dataset_config.label:
            fetched_labels = self._fetch(
                self.dataset_config.label.input_feature)
            if fetched_labels is not None:
                labels = fetched_labels[:]

        if labels is not None:
            label_index = {}

            for idx, label in enumerate(labels):
                if label not in label_index:
                    label_index[label] = []
                label_index[label].append(idx)
            self.unique_class_ids = list(set(label_index.keys()))

            labels = [k for k in label_index.keys()]
            self.num_classes = len(labels)

            self.class_index = []
            self.class_sizes = []
            for idx, label in enumerate(labels):
                ls = label_index[label]
                self.class_index.append(ls)
                self.class_sizes.append(len(ls))

    def get_input_shape(self):
        input_shapes = {}
        datum = self.example(0)
        for name, value in six.iteritems(datum):
            input_shapes[name] = np.shape(value)
        return input_shapes

    def __len__(self):
        return self.length

    def _fetch(self, name, idx=None, default=None):
        if name not in self.data:
            return default
        if idx is not None:
            return self.data[name][idx]
        return self.data[name]

    def example(self, idx, augment=False, preprocess=True):
        features = {}
        for output_feature_name, input_feature_name in six.iteritems(
                self.dataset_config.input_features):
            value = self._fetch(input_feature_name, idx)
            features[input_feature_name] = value

        if augment or preprocess:
            features = self.dataset_config.transform(
                features, augment=augment, preprocess=preprocess)

        return features

    def id(self, idx):
        if not self.dataset_config.id:
            return "Unknown"

        value = self._fetch(
            self.dataset_config.id.input_feature, idx, default="Unknown")
        return self.dataset_config.transforms["__ID__"].transform(value)

    def label(self, idx=None):
        if self.dataset_config.label:
            value = self._fetch(
                self.dataset_config.label.input_feature,
                idx,
                default="Unknown")
        else:
            return None
        return self.dataset_config.transforms["__LABEL__"].transform(value)

    def metadata(self, idx):
        if not self.dataset_config.metadata:
            return {}

        value = self._fetch(
            self.dataset_config.metadata.input_feature, idx, default={})
        return self.dataset_config.transforms["__METADATA__"].transform(value)

    def group(self, idx):
        if not self.dataset_config.group:
            return "Unknown"

        value = self._fetch(
            self.dataset_config.group.input_feature, idx, default="Unknown")
        return self.dataset_config.transforms["__GROUP__"].transform(value)

    def __getitem__(self, idx):
        return Datum(idx, self.id(idx),
                     self.example(idx, augment=True, preprocess=True),
                     self.example(idx, augment=False, preprocess=False),
                     self.label(idx), self.metadata(idx), self.group(idx))

    def get_item(self, idx, augment=False, preprocess=True):
        return Datum(idx, self.id(idx),
                     self.example(idx, augment=augment, preprocess=preprocess),
                     self.example(idx, augment=False, preprocess=False),
                     self.label(idx), self.metadata(idx), self.group(idx))

    def get_augmented_item(self, idx):
        return Datum(idx, self.id(idx),
                     self.example(idx, augment=True, preprocess=True),
                     self.example(idx, augment=False, preprocess=False),
                     self.label(idx), self.metadata(idx), self.group(idx))

    def num_classes(self):
        return self.num_classes

    def random_classes(self, N=3):
        """Randomly select N class indices from the dataset, without
        replacement.

        Note that N should typically be small, relative to the size of your
        dataset. If for some reason you need a large N, you may be better
        off with a `np.random.randint(self.num_classes, N, replace=False)`

        Keyword Arguments:
            N {int} -- Number of classes to randomly select (default: {3})

        Returns:
            [list(int)] -- A list of class indices selected.
        """

        class_indexes = []

        # Effectively: we choose the first element at random from #classes.
        # Then, we choose the next element by choosing a random number from
        # 0 - #classes-1, and manually skip over the elements we've already
        # chosen.
        for i in range(N):
            idx = rng.randint(0, self.num_classes - i)
            o = 0
            for cls in class_indexes:
                if cls <= idx:
                    o += 1
            idx = idx + o
            class_indexes.append(idx)

        return class_indexes

    def all_labels(self):
        return self.unique_class_ids

    def random_idx_for_class(self, class_idx):
        """Randomly select an item index for a particular class.

        Arguments:
            class_idx {int} -- Index of the class.

        Returns:
            int -- The index of a randomly selected item from the dataset.
        """
        return rng.choice(self.class_index[class_idx])


class Transformation(object):
    """Representation of a transformation of a raw feature to a processed
    feature.
    """

    def __init__(
            self,
            input_feature=None,
            # Old name for input_feature
            feature_name=None,
            augmentation=None,
            preprocessing=None):
        self.input_feature = input_feature
        if not input_feature and feature_name:
            self.input_feature = feature_name
        self.augmentation = deserialize_moirai_object(augmentation)
        self.preprocessing = deserialize_moirai_object(preprocessing)

    def transform(self, value, augment=False, preprocess=True):
        if augment and self.augmentation:
            value = self.augmentation(value)

        if preprocess and self.preprocessing:

            value = self.preprocessing(value)

        return value

    def __repr__(self):
        return str(self.get_config())

    def get_config(self):
        return {
            "input_feature":
            self.input_feature,
            "augmentation":
            self.augmentation.get_config() if self.augmentation else None,
            "preprocessing":
            self.preprocessing.get_config() if self.preprocessing else None
        }


register_custom_object("Transformation", Transformation)


class DatasetConfig(object):
    def __init__(self,
                 name="examples",
                 features=[],
                 id=None,
                 label=None,
                 group=None,
                 metadata=None):
        self.name = name
        self.id = self.deserialize_transform(id)
        self.label = self.deserialize_transform(label)
        self.metadata = self.deserialize_transform(metadata)
        self.group = self.deserialize_transform(group)

        self.input_features = {}
        self.transforms = {}
        # Feature names MUST be provided in the order the inputs are declared
        # in the model.
        self.ordered_output_features = []

        for output_feature, config in features:
            self.transforms[output_feature] = self.deserialize_transform(
                config)
            self.input_features[output_feature] = self.transforms[
                output_feature].input_feature
            self.ordered_output_features.append(output_feature)

        if self.id:
            self.transforms["__ID__"] = self.deserialize_transform(self.id)

        if self.label:
            self.transforms["__LABEL__"] = self.deserialize_transform(
                self.label)

        if self.metadata:
            self.transforms["__METADATA__"] = self.deserialize_transform(
                self.metadata)

        if self.group:
            self.transforms["__GROUP__"] = self.deserialize_transform(
                self.group)

    def deserialize_transform(self, transform):
        if isinstance(transform, dict) and 'type' in transform:
            return json_dict_to_moirai_obj(transform)

        name_or_obj = deserialize_moirai_object(transform)
        if isinstance(transform, six.string_types):
            return Transformation(input_feature=name_or_obj)
        return name_or_obj

    def all_features(self):
        return self.all_features

    def transform(self, input_dict, augment=False, preprocess=True):
        output = {}

        for output_feature_name, transform in six.iteritems(self.transforms):
            input_feature_name = transform.input_feature
            if preprocess:
                output_name = output_feature_name
            else:
                output_name = input_feature_name

            if input_feature_name in input_dict:
                value = input_dict[input_feature_name]
                transformed_value = transform.transform(
                    value, augment=augment, preprocess=preprocess)
                output[output_name] = transformed_value

        return output

    def get_config(self):
        transforms = []
        for output_feature_name in self.ordered_output_features:
            transform = self.transforms[output_feature_name]
            if transform is not None:
                transforms.append((output_feature_name,
                                   serialize_moirai_object(transform)))

        return {
            "name": self.name,
            "id": serialize_moirai_object(self.id),
            "label": serialize_moirai_object(self.label),
            "metadata": serialize_moirai_object(self.metadata),
            "group": serialize_moirai_object(self.group),
            "features": transforms
        }


register_custom_object("DatasetConfig", DatasetConfig)
