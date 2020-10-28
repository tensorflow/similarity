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

import copy
import json
import multiprocessing
import tempfile
import warnings

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import deserialize_keras_object as deserialize
from tensorflow.keras.utils import serialize_keras_object as serialize

from tensorflow_similarity.api.callbacks import OverrideModelCallback
from tensorflow_similarity.api.engine.augmentation import Augmentation
from tensorflow_similarity.api.engine.database import Database
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.simhash import SimHashInterface
from tensorflow_similarity.api.engine.task import MetaTask, Task
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask
from tensorflow_similarity.api.tasks.inference import InferenceTask
from tensorflow_similarity.api.tasks.prewarm_task import create_prewarming_task
from tensorflow_similarity.api.tasks.quadruplet_loss_task import QuadrupletLossTask
from tensorflow_similarity.callbacks.hard_mining import ResultWriter
from tensorflow_similarity.callbacks.validation_set_metrics import ValidationCallback
from tensorflow_similarity.utils.config_utils import register_custom_object
from tensorflow_similarity.utils.model_utils import (
    clone_model_inputs,
    example_list_to_training_data,
    training_data_to_example_list)


class SimilarityModel(SimHashInterface, Model):
    def __init__(self,
                 tower_model=None,
                 hard_mining=False,
                 auxillary_tasks=[],
                 hard_mining_directory=None,
                 strategy="quadruplet_loss",
                 preprocessing=None,
                 augmentation=None,
                 optimizer=None,
                 name=None,
                 towers=[],
                 database=None,

                 # ! Not used, but will be passed by model_from_config.
                 layers=[],
                 input_layers=[],
                 output_layers=[],
                 **generator_config):
        if not isinstance(tower_model, Model):
            tower_model = Model.from_config(tower_model)

        cloned_input_names, cloned_input_tensors = clone_model_inputs(
            tower_model, "")

        tower_output = tower_model(cloned_input_tensors)

        # Initialize the model based on the tower_model's inputs and outputs.
        # For all intents and purposes, this _IS_ the tower model, but with
        # special handling for things like .fit()
        Model.__init__(self,
                       cloned_input_tensors,
                       tower_output,
                       name=name)

        self.tower_model = tower_model
        self.towers = towers
        self.hard_mining = hard_mining
        if not hard_mining_directory:
            hard_mining_directory = tempfile.mkdtemp()
        self.hard_mining_directory = hard_mining_directory
        self.strategy = strategy
        self.generator_config = generator_config

        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.database = database
        if database and not isinstance(database, Database):
            self.database = deserialize(database)

        # Our tasks have generators, which require the example/label data to
        # create. Defer execution until fit is called.
        #
        # TODO - find a way to defer creation of the generators until later,
        # and instantiate the tasks
        # now for fail-fast purposes.
        self.training_task = None
        self.prewarm_task = None
        # Inference task is only dependent on the tower model, so we can create
        # the task here.
        # ! This may look out of place, but as the task is only dependent on
        # ! the inference task, it's possible to do here. Deferring creation
        # ! to fit() does not make sense, as a sequence of:
        # ! load_model() -> predict() is common.
        self.inference_task = self._build_inference_task()

        if not optimizer:
            optimizer = Adam(lr=0.001)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            if isinstance(optimizer, str):
                optimizer = tf.keras.optimizers.get(optimizer)
            else:
                optimizer = tf.keras.optimizers.deserialize(optimizer)

        # ! This can't just be called 'optimizer' because it will conflict with
        # ! the version stored on the `Model`, which would have extra
        # ! attributes present when the model is compiled.
        self.raw_optimizer = optimizer

        if not isinstance(preprocessing, Preprocessing):
            preprocessing = deserialize(preprocessing)
        if not isinstance(augmentation, Augmentation):
            augmentation = deserialize(augmentation)
        self.generator_config["preprocessing"] = preprocessing
        self.generator_config["augmentation"] = augmentation

        o_auxillary_tasks = []
        for task in auxillary_tasks:
            if not isinstance(task, Task):
                task = deserialize(task)

            o_auxillary_tasks.append(task)
        self.auxillary_tasks = o_auxillary_tasks

        self.compile(self.raw_optimizer or Adam(.001), "mae")

    def inference_model(self):
        if not self.inference_task:
            return None
        return self.inference_task.task_model

    @classmethod
    def from_config(cls, config):
        """Override from_config to pass all configuration parameters to the
        model."""
        return cls(**config)

    def get_config(self):
        aux_tasks_config = []
        for task in self.auxillary_tasks:
            aux_tasks_config.append(serialize(task))

        serialized_optimizer = serialize(self.raw_optimizer)

        database_config = None
        if self.database:
            database_config = serialize(self.database)

        config = self.tower_model.get_config()
        config.update({
            "tower_model": self.tower_model.get_config(),
            "hard_mining": self.hard_mining,
            "auxillary_tasks": aux_tasks_config,
            "hard_mining_directory": self.hard_mining_directory,
            "strategy": self.strategy,
            "preprocessing": serialize(self.preprocessing),
            "augmentation": serialize(self.augmentation),
            "database": database_config
        })

        for name, value in self.generator_config.items():
            # These are in the top-level config, making them redundant here.
            if name == "preprocessing" or name == "augmentation":
                continue

            if hasattr(value, "get_config"):
                value = serialize(value)
            config[name] = value

        return config

    def build_database(self, x, y, **kwargs):
        emb = self.predict(x, **kwargs)
        self.set_database(Database(emb, y))
        return self.database

    def get_database(self):
        return self.database

    def _build_auxillary_tasks(self, x, y):
        return self.auxillary_tasks

    def _build_training_task(self, x, y):
        raise NotImplementedError()

    def _build_prewarm_task(self):
        prewarm = create_prewarming_task(self.training_task)
        prewarm.build(compile=True)
        return prewarm

    def _build_inference_task(self):
        inference = InferenceTask("inference",
                                  self.tower_model,
                                  preprocessing=self.preprocessing)
        inference.build(compile=True)
        return inference

    def _build_tasks(self, x, y, include_prewarm):
        self.training_task = self._build_training_task(x, y)

        if include_prewarm and len(self.training_task.auxillary_tasks):
            self.prewarm_task = self._build_prewarm_task()
        else:
            self.prewarm_task = None
        self.inference_task = self._build_inference_task()

    def _move_special_callbacks_to_end(self, callbacks):
        """Move callbacks that cannot handle our pseudo-metrics to the end
        of the list, after the pseudo-metrics have been removed from the
        logs.

        Arguments:
            callbacks {list} -- List of keras `Callback`s.

        Returns:
            list -- List of keras `Callback`s, after changing the order.
        """
        main_callbacks = []
        moved_callbacks = []

        incompatible_callbacks = [
            'MonitorCallback',
            'DisplayCallback',
            'TensorBoard'
        ]

        for callback in callbacks:
            callback_clsname = callback.__class__.__name__
            if callback_clsname in incompatible_callbacks:
                moved_callbacks.append(callback)
            else:
                main_callbacks.append(callback)

        return main_callbacks + moved_callbacks

    def _sanity_check_validation_data(self, validation_data):
        x, y = validation_data

        assert isinstance(x, dict)
        assert isinstance(y, dict)

        assert "targets" in x, x.keys()
        assert "targets" in y, y.keys()

        assert isinstance(x["targets"], dict)
        assert len(x) > 1

    def _common_callbacks(self):
        callbacks = [
            ResultWriter(hard_mining_directory=self.hard_mining_directory,
                         towers=self.towers,
                         write_embeddings=False)
        ]
        return callbacks

    def _prepare_training_callbacks(self, fit_callbacks):
        training_callbacks = fit_callbacks + self.training_task.get_callbacks()
        training_callbacks = training_callbacks + self._common_callbacks()
        training_callbacks = self._move_special_callbacks_to_end(
            training_callbacks)

        model_override_callback = OverrideModelCallback(
            self, training_callbacks)
        # Run the model override callback first.
        training_callbacks = [model_override_callback] + training_callbacks

        return training_callbacks

    def _prepare_prewarm_callbacks(self, fit_callbacks, prewarm_epochs):
        prewarm_callbacks = []
        if prewarm_epochs and self.prewarm_task:
            additional_prewarm_callbacks = self.prewarm_task.get_callbacks()
            prewarm_callbacks = fit_callbacks + additional_prewarm_callbacks
            prewarm_callbacks = self._move_special_callbacks_to_end(
                prewarm_callbacks)
        return prewarm_callbacks

    def fit(self,
            x,
            y,
            prewarm_epochs=0,
            epochs=100,
            generator_workers=None,
            callbacks=[],
            preprocess_validation=False,
            similarity_validation_data=None,
            validation_data=None,
            validation_split=None,
            **kwargs):
        """Custom fit method.

        Arguments:
            x {dict} -- Dictionary of feature name to np.array containing the
                 feature data.
            y {dict} -- Dictionary of label name to np.array containing the
                 label data.

        Keyword Arguments:
            prewarm_epochs {int} -- Number of epochs to run the auxillary
                tasks before starting the full similarity model. (default: {0})
            epochs {int} -- Number of epochs to train the model, after the
                prewarm epochs, if
               any complete. Total epochs for the model is thus:
                   (prewarm_epochs + epochs) (default: {100})
            generator_workers {int} -- Number of workers to use to generate the
                tuple data. If None, the main thread will generate tuples.
                (default: {None})
            callbacks {list} -- List of Callbacks to use during pre-warm and
                training. Note that the Similarity system will add other
                callbacks (e.g. a callback to write hard tuples) (default: {[]})

        Raises:
            ValueError: If incompatible arguments are provided (e.g.
                prewarm_epochs > 0 and no auxillary tasks are provided)

        Returns:
            [type] -- [description]
        """
        assert validation_split is None, "SimilarityModel.fit() does not " \
            "support the validation_split keyword argument, as the " \
            "format for similarity validation data is not the same " \
            "as the format for the training data."
        assert validation_data is None, "SimilarityModel.fit() does not " \
            "support the validation_data keyword argument, as the " \
            "format for similarity validation data is not the same " \
            "as the format for normal validation. Use" \
            "similarity_validation_data=... instead"

        if generator_workers is None:
            generator_workers = max(1, multiprocessing.cpu_count() - 2)

        # Build tasks - we had to defer this until now, as we need the x, y
        # data to generate the generators, which in turn are needed to generate
        # the tasks.
        self._build_tasks(x, y, include_prewarm=prewarm_epochs > 0)

        common_callbacks = self._common_callbacks()

        validation_data = self._prepare_validation_set(
            similarity_validation_data, preprocess_validation)
        if validation_data:
            validation_callback = self._make_validation_callback(
                validation_data)
            common_callbacks.append(validation_callback)

        training_callbacks = self._prepare_training_callbacks(callbacks)
        prewarm_callbacks = self._prepare_prewarm_callbacks(
            callbacks, prewarm_epochs)

        with warnings.catch_warnings():
            # Validation can be slow, but there's no need to spam about it.
            warnings.simplefilter("ignore", UserWarning)

            if not prewarm_epochs and not epochs:
                raise ValueError("Requested to prewarm for 0 epochs and "
                                 "train for 0 epochs.")

            if self.prewarm_task is not None:
                print("Pre-warming the model for %d epochs." % prewarm_epochs)

                results = self.prewarm_task.task_model.fit_generator(
                    self.prewarm_task.generator,
                    epochs=prewarm_epochs,
                    callbacks=prewarm_callbacks,
                    use_multiprocessing=generator_workers > 1,
                    max_queue_size=(generator_workers + 1) * 16,
                    workers=generator_workers if generator_workers > 1 else 0,
                    **kwargs)
                print("Pre-warm complete. Beginning main training.")

            elif prewarm_epochs:
                raise ValueError(
                    "Warning: prewarm_epochs was non-zero, but there are no "
                    "auxillary tasks, and nothing to prewarm.")

            if epochs > 0:
                results = self.training_task.task_model.fit_generator(
                    self.training_task.generator,
                    epochs=prewarm_epochs + epochs,
                    callbacks=training_callbacks,
                    use_multiprocessing=generator_workers > 1,
                    max_queue_size=(generator_workers + 1) * 16,
                    workers=generator_workers if generator_workers > 1 else 0,
                    initial_epoch=prewarm_epochs,
                    **kwargs)
            return results

    def predict(self, x, output_dict=False, **kwargs):
        res = self.inference_task.predict(x, **kwargs)
        if output_dict:
            res = self.inference_task.interpret_predictions(res)
        return res

    def neighbors(self, x, N=50):
        assert self.database is not None
        embeddings = self.inference_task.task_model.predict(x)
        return self.database.query(embeddings, N=N)

    def save_model(self, filename, overwrite=True):
        self.save(filename)

    @classmethod
    def load_model(cls, filename):
        return tf.keras.models.load(filename)

    def evaluate(self, x_test, y_test, x_targets=None, y_targets=None):
        if self.preprocessing and False:
            x_test = self.preprocessing.prepocess(x_test)
            x_targets = self.preprocessing.prepocess(x_targets)

        return super(SimilarityModel, self).evaluate(x_test,
                                                     y_test,
                                                     x_targets=x_targets,
                                                     y_targets=y_targets)

    def _prepare_validation_set(self, validation_data, preprocess_validation):
        if not validation_data:
            return None

        if not preprocess_validation:
            return validation_data

        if not self.preprocessing:
            raise ValueError(
                "preprocess_validation=True in fit(), but there no preprocessor was set.")

        # Preprocess the validation data
        (x, y) = validation_data

        x_out = {}
        for dataset_name, dataset in x.items():
            examples = training_data_to_example_list(dataset)
            pp_examples = [self.preprocessing(x) for x in examples]
            preprocessed = example_list_to_training_data(pp_examples)
            x_out[dataset_name] = preprocessed

        return (x_out, y)

    def _make_validation_callback(self, validation_data):
        self._sanity_check_validation_data(validation_data)
        x, y = validation_data
        return ValidationCallback(x, y)


register_custom_object("SimilarityModel", SimilarityModel)
