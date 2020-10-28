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

from tensorflow_similarity.callbacks.base import MoiraiCallback


class MetricsCallbackPlugin:
    """A plugin that computes some metric based on the embeddings, database, or
        neighbors of the provided test set.

        This class should be extended for any callback that need to do
            computation on test and/or target embeddings.

    Arguments:
        frequency (int): Frequency (in epochs) at which compute_metrics will
            be performed. Default to 1.
    """

    def __init__(self, frequency=1):
        assert isinstance(
            frequency, int), "frequency is not integer: %r" % frequency
        assert frequency >= 1, "frequency is less than 1: %r" % frequency

        self.frequency = frequency

    def compute_metrics(self,
                        simhash,
                        database,
                        neighbors,
                        embedding_test,
                        x_test,
                        y_test,
                        embedding_targets,
                        x_targets,
                        y_targets,
                        epoch,
                        logs={}):
        """Compute metrics to be overwritten by custom metrics plugins.

        Arguments:
            simhash (SimHashInterface): SimHash object,
                used for training/inference.
            database (dict: validation set -> Database): Per-dataset database
                (searchable mapping of embedding -> label) of targets.
            neighbors (dict: validation set -> list[LabeledNeighbor]):
                Per-dataset, per-example exhaustive lists of nearest neighbors.
            embedding_test (dict: validation set -> list[embeddings]):
                Per-dataset test embeddings.
            x_test (dict: validation set -> list[???]):
                Per-dataset test examples.
            y_test (dict: validation set -> list[int]):
                Per-dataset test labels.
            embedding_targets (dict: validation set -> list[embeddings]):
                Per-dataset target embeddings.
            x_targets (dict: validation set -> list[???]):
                Per-dataset target examples.
            y_targets (dict: validation set -> list[int]):
                Per-dataset target labels.
            epoch (int): Current epoch, from the Keras callback.

        Keyword Arguments:
            logs (dict): Current logs, from the Tensorflow Keras callback.
        """
        pass

    def on_train_end(self, logs=None):
        """The plugin method to be called at the end of training.

        Arguments:
            logs (dict): Current logs, from the Tensorflow Keras callback.
        """
        pass


class MetricsCallback(MoiraiCallback):
    def __init__(self, plugins, x_test, y_test, x_targets, y_targets):
        """MetricsCallback wraps one or more metrics callback plugins.

        Arguments:
            plugins (list: MetricsCallbackPlugin): Callbacks that will be
                called with the current embeddings of the test and target sets.
            x_test (object): Representation of the test set to embed.
                Either:
                    dict: validation_set_name -> training_data   OR
                    object: training_data (list of examples or
                        dictionary of features)

            y_test (object): Labels for the test set
                Either:
                    dict: validation_set_name -> labels   OR
                    list: labels

            x_targets (object): Representation of the target points, which
                the test set will be compared to.
                Either:
                    dict: validation_set_name -> training_data   OR
                    object: training_data (list of examples or
                        dictionary of features)

            y_targets (object): Labels for the target points
                Either:
                    dict: validation_set_name -> labels   OR
                    list: labels

        """

        self.plugins = plugins
        self.x_test = x_test
        self.y_test = y_test
        self.x_targets = x_targets
        self.y_targets = y_targets

    def on_epoch_end(self, epoch, logs={}):

        database = self.simhash.build_database(self.x_targets, self.y_targets)
        embedding_targets = database.get_embeddings()

        # Get embeddings for the test set.
        embedding_test = self.simhash.predict(self.x_test)

        # Query the built database, to find the nearest targets.
        neighbors = database.query(embedding_test, N=len(self.y_targets))

        for plugin in self.plugins:
            # compute each plugin in an interval set by plugin.frequency
            if epoch % plugin.frequency == 0:
                plugin.compute_metrics(self.simhash,
                                       database,
                                       neighbors,
                                       embedding_test,
                                       self.x_test,
                                       self.y_test,
                                       embedding_targets,
                                       self.x_targets,
                                       self.y_targets,
                                       epoch,
                                       logs)

    def on_train_end(self, logs=None):
        for plugin in self.plugins:
            plugin.on_train_end(logs)
