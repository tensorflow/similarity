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

from tensorflow_similarity.api.metrics.common_metrics import CommonMetrics
from tensorflow_similarity.api.engine.database import Database
import six


class SimHashInterface(object):
    """SimHashInterface is the underlying interface that the SimHash wrapper,
    and all implementations are responsible for implementing.
    """

    def fit(self, x, y, **kwargs):
        """Fits this model to the given dataset.

        # Arguments
        x: Numpy array of training data (if the model has a single input),
            or list of Numpy arrays (if the model has multiple inputs).
            If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
            `x` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        y: Numpy array of target (label) data
            (if the model has a single output),
            or list of Numpy arrays (if the model has multiple outputs).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
            `y` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        **kwargs: Passed directly to the underlying model implementation.
        """
        raise NotImplementedError

    def predict(self, x, **kwargs):
        """Applies this model to the given dataset.

        # Arguments
        x: Numpy array of training data (if the model has a single input),
            or list of Numpy arrays (if the model has multiple inputs).
            If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
            `x` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        **kwargs: Passed directly to the underlying model implementation.

        # Returns
        A Numpy array containing the embeddings for the dataset.
        """
        raise NotImplementedError

    def evaluate(self, x_test, y_test, x_targets=None, y_targets=None):
        """Evaluate the given dataset(s).

        (x|y)_test are the items to be infered.
        (x|y)_targets are the neighborhood to use for ranking metrics. If
            unspecified, (x|y)_test will be used.
        """

        if x_targets is None:
            x_targets = x_test
            y_targets = y_test

        e_targets = self.predict(x_targets)
        e_test = self.predict(x_test)

        output_metrics = {}

        metrics = CommonMetrics(x_test, y_test, e_test, x_targets, y_targets,
                                e_targets).compute()
        for k, v in six.iteritems(metrics):
            output_metrics[k] = v.item()

        return output_metrics

    def build_database(self, x, y, **kwargs):
        """Builds a database for querying nearest neighbors. Given an input
        example set, and their labels, builds a Database object.

        x: Numpy array of training data (if the model has a single input),
            or list of Numpy arrays (if the model has multiple inputs).
            If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
            `x` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        y: Numpy array of target (label) data
            (if the model has a single output),
            or list of Numpy arrays (if the model has multiple outputs).
            If output layers in the model are named, you can also pass a
            dictionary mapping output names to Numpy arrays.
            `y` can be `None` (default) if feeding from
            framework-native tensors (e.g. TensorFlow data tensors).
        **kwargs: Passed directly to the underlying model implementation's
            predict function.
        """
        raise NotImplementedError("not implemented by %s" %
                                  self.__class__.__name__)

    def get_weights(self):
        """Returns the current weights for the model.

        # Returns
            The weights values as a list of numpy arrays.
        """
        raise NotImplementedError

    def set_weights(self, weights):
        """Sets the weights of the layer, from Numpy arrays.
        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        # Raises
            ValueError: If the provided weights list does not match the
                layer's specifications.
        """
        raise NotImplementedError

    def get_database(self):
        """Returns the current database, or None if a database has not been
        built."""
        return self.database

    def set_database(self, db):
        """Sets the current database."""
        self.database = db

    def get_config(self):
        return {}

    def save_model(self, filename):
        raise NotImplementedError

    def load_model(self, filename):
        raise NotImplementedError


class SimHashRegistry(object):
    _STRATEGIES = {}
    @classmethod
    def register(cls, name, strategy_class):
        """Register a strategy name/class, so that it can be referred to by name.

        Example:
            >>> class MySimHashImpl(object):
            ...     pass
            ... SimHashRegistry.register("mysimhash", MySimHashImpl)
            ... simhash = SimHash(tower_model, strategy="mysimhash")
            ... assert isinstance(simhash.model, MySimHashImpl)
        """
        SimHashRegistry._STRATEGIES[name] = strategy_class

    @classmethod
    def contains(cls, name):
        return name in SimHashRegistry._STRATEGIES


def SimHash(*args, strategy="hard_quadruplets", **kwargs):
    if not SimHashRegistry.contains(strategy):
        valid_strategies = ", ".join(SimHashRegistry._STRATEGIES)
        raise ValueError(
            "Unknown strategy: '{0}' - known strategies are: {1}".format(
                strategy, valid_strategies))
    return SimHashRegistry._STRATEGIES[strategy](*args, **kwargs)
