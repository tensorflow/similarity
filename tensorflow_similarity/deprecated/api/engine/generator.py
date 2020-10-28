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

import numpy as np
import six
from tensorflow.keras.utils import Sequence


class Generator(Sequence):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, batch_id):
        """ Support standard array notation, to fulfill the contract of tf.keras.utils.Sequence.

        Retrieves the Batch object and transforms it into the (example, label) format expected by Keras.

        Args:
            batch_id (int): The batch id.

        Returns:
            A dictionary of examples, and a dictionary of labels.
        """

        batch = self.get_batch(batch_id)
        return batch.values, batch.labels

    def get_batch(self, batch_id):
        """Get the `batch_id`th batch, as a Batch object.

        Args:
            batch_id (int): The id of the batch to retrieve (between 0 and len(self)).

        Raises:
            NotImplementedError: This method is meant to be implemented by subclasses.
        """
        raise NotImplementedError


class GeneratorWrapper(object):
    def __init__(self, generator_class, **generator_config):
        self.generator_class = generator_class
        self.generator_config = generator_config

    def get(self, x, y):
        return self.generator_class(x, y, **self.generator_config)

    def get_config(self):
        return self.generator_config


Element = collections.namedtuple("Element", [
    'index',
    'feature_dictionary',
    'preprocessed_dictionary',
    'raw_feature_dictionary',
    'label_id',
    'augmented'
])


class Batch(object):
    def __init__(self):
        self.labels = {}
        self.raw_values = {}
        self.preprocessed_raw_values = {}
        self.values = {}

        self.feature_sources = {}

    def _source_key(self, bucket_name, name):
        return "%s:%s" % (bucket_name, name)

    def sanity_check_data(self, souce, bucket_name, name):
        key = self._source_key(bucket_name, name)

        previous_source = self.feature_sources.get(key, None)
        if previous_source:
            raise ValueError(
                "souce '%s' is attempting to add '%s' to %s, however it "
                "conflicts with a value from the '%s' souce" %
                (souce, name, bucket_name, previous_source))

    def set_source(self, souce, bucket_name, name):
        self.sanity_check_data(souce, bucket_name, name)
        key = self._source_key(bucket_name, name)
        self.feature_sources[key] = souce

    def _add_dict(self, souce, from_dictionary, bucket_name, to_dictionary):
        for k, v in six.iteritems(from_dictionary):
            if not isinstance(v, np.ndarray):
                v = np.array(v)

            self.set_source(souce, bucket_name, k)
            to_dictionary[k] = v

    def add_raw_features(self, source, feature_dictionary):
        self._add_dict(source, feature_dictionary, "raw_values",
                       self.raw_values)

    def add_preprocessed_raw_features(self, source, feature_dictionary):
        self._add_dict(source, feature_dictionary, "preprocessed_raw_values",
                       self.preprocessed_raw_values)

    def add_features(self, source, feature_dictionary):
        self._add_dict(source, feature_dictionary, "values", self.values)

    def add_labels(self, source, label_dictionary):
        self._add_dict(source, label_dictionary, "labels", self.labels)

    def _update_dict(self, souce, from_dictionary, bucket_name, to_dictionary):
        npified_dict = {}
        for k, v in six.iteritems(from_dictionary):
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            npified_dict[k] = v

        to_dictionary.update(npified_dict)

    def update_raw_features(self, source, feature_dictionary):
        self._update_dict(source, feature_dictionary, "raw_values",
                          self.raw_values)

    def update_preprocessed_raw_features(self, source, feature_dictionary):
        self._update_dict(
            source,
            feature_dictionary,
            "preprocessed_raw_values",
            self.preprocessed_raw_values)

    def update_features(self, source, feature_dictionary):
        self._update_dict(source, feature_dictionary, "values", self.values)

    def update_labels(self, source, label_dictionary):
        self._update_dict(source, label_dictionary, "labels", self.labels)

    def get(self, name, val_type="augmented"):
        """Gets a value from this batch.

        Arguments:
            name {str} -- name of the value to retrieve.

        Keyword Arguments:
            val_type {str} -- One of "augmented", "preprocessed", or "raw".
                (default: {"augmented"})
        """
        if val_type == "augmented":
            return self.values[name]
        if val_type == "preprocessed":
            return self.preprocessed_raw_values[name]
        if val_type == "raw":
            return self.raw_values[name]
        raise ValueError("Unknown val_type='%s'" % val_type)

    def merge(self, source, other_batch):
        self.add_raw_features(source, other_batch.raw_values)
        self.add_features(source, other_batch.values)
        self.add_labels(source, other_batch.labels)
