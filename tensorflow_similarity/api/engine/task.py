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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import serialize_keras_object as serialize

from tensorflow_similarity.api.callbacks.results_callbacks import RemoveSuppressedMetrics
from tensorflow_similarity.api.engine.metrics import Pseudometric
from tensorflow_similarity.api.generators.task_based_generator import TaskBasedGenerator
from tensorflow_similarity.api.losses.no_loss import NoLoss
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils import model_utils


class Task(object):
    """An auxillary task is a separate task that is trained along side the
    normal embedding task. This could be an autoencoder, a supervised
    classification problem, or any other task that can make use of the
    embedding being produced.

    AuxillaryTasks interact with the system in a few places:
    1. Generator - each auxillary task will have an output (or set of outputs),
    and require a label (or set of labels)
    2. Strategy - each task needs to augment the model, to insert its own
    logic for computing the answer.
    3. Loss functions - each task will provide a loss function (or dictionary
    of loss functions if there are multiple outputs)
    """

    def __init__(self, name, tower_model):
        """Creates the task

        Arguments:
            tower_model {tf.keras.models.Model} -- The model applied to
            input_layer_dictionary {[string] -> [tf.keras.layers.InputLayer]} -- dictionary
                mapping input layer names to InputLayers.
        """
        self.name = name
        self.tower_model = tower_model
        self.task_losses = {}
        self.task_loss_weights = {}
        self.task_inputs = []
        self.task_input_dict = {}
        self.task_outputs = []
        self.task_input_names = []
        self.task_output_names = []
        self.task_model = None
        self.task_metrics = {}
        self.built = False
        self.suppressed_metric_prefixes = []
        self.callbacks = []

    def build_task(self):
        raise NotImplementedError()

    def get_input_layers(self):
        """
        Returns the list of input layers for the task.

        Layers will be returned in the same order as get_input_names().
        """
        return self.task_inputs

    def get_input_names(self):
        """
        Returns the list of input names for the task.
        Layer names will be returned in the same order as get_input_layers().
        """
        return self.task_input_names

    def get_input_dictionary(self):
        return self.task_input_dict

    def get_output_layers(self):
        return self.task_outputs

    def get_output_names(self):
        return self.task_output_names

    def get_output_dict(self):
        return zip(self.task_output_names, self.task_outputs)

    def get_loss_dict(self):
        return self.task_losses

    def get_loss_weight_dict(self):
        return self.task_loss_weights

    def _add_input(self, name, layer):
        self.task_inputs.append(layer)
        self.task_input_names.append(name)
        self.task_input_dict[name] = layer

    def _add_output(self, name, layer, loss, loss_weight=1.0, metric=None):
        self.task_outputs.append(layer)
        self.task_output_names.append(name)
        self.task_losses[name] = loss
        self.task_loss_weights[name] = loss_weight

        if metric:
            self.task_metrics[name] = metric

        # If the loss is a no-op, there's no need to clutter up the display.
        if isinstance(loss, NoLoss):
            self.suppressed_metric_prefixes.append(name + "_loss")

    def _add_callback(self, callback):
        self.callbacks.append(callback)

    def get_callbacks(self):
        if not self.built:
            raise ValueError(
                "task.build() must be called before task.get_callbacks()")

        if len(self.suppressed_metric_prefixes):
            suppression_callback = RemoveSuppressedMetrics(
                self.suppressed_metric_prefixes)
            return self.callbacks + [suppression_callback]

        return self.callbacks

    def get_metrics(self):
        return self.task_metrics

    def build(self, compile=False):
        if self.built:
            return
        self.build_task()

        # TODO(b/141715442): experimental_run_tf_function=False is a workaround
        # for b/141715442.
        if compile:
            self.task_model.compile(
                optimizer="adam",
                loss=self.task_losses,
                loss_weights=self.task_loss_weights,
                metrics=self.task_metrics,
                experimental_run_tf_function=False)
        self.built = True

    def get_model(self):
        return self.task_model

    def interpret_predictions(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]
        out_names = self.get_output_names()

        num_outputs = len(outputs)

        assert num_outputs == len(out_names)

        out = {}
        for idx, name in enumerate(out_names):
            out[name] = outputs[idx]
        return out

    def get_config(self):
        return {
            "name": self.name,
            "tower_model": serialize(self.tower_model)
        }


class AuxillaryTask(Task):
    def __init__(self, name, tower_model=None):
        super(AuxillaryTask, self).__init__(name, tower_model=None)

    def update_batch(self, batch):
        raise NotImplementedError()


class MainTask(Task):
    def __init__(self, name, generator, tower_model):
        super(MainTask, self).__init__(name, tower_model)
        self.generator = generator

    def get_generator(self):
        return self.generator

    def get_main_batch(self, seq_id):
        return self.generator.get_batch(seq_id)


class MetaTask(MainTask):
    def __init__(self,
                 name,
                 tower_model,
                 main_task,
                 auxillary_tasks=[],
                 inference_task=None,
                 optimizer=None):
        generator = TaskBasedGenerator(main_task, auxillary_tasks)
        super(MetaTask, self).__init__(name, generator, tower_model)

        self.inference_task = inference_task
        self.main_task = main_task
        self.auxillary_tasks = auxillary_tasks

        self.optimizer = optimizer
        if not optimizer:
            self.optimizer = tf.keras.optimizers.Adam(lr=.001)

    def get_config(self):
        config = {}

        aux_task_config = []
        for task in self.auxillary_tasks:
            aux_task_config.append(serialize(task))

        return {
            "inference_task": serialize(self.inference_task),
            "main_task": serialize(self.main_task),
            "auxillary_tasks": aux_task_config,
            "optimizer": tf.keras.optimizers.serialize(self.optimizer)
        }

    def build(self, compile=False, show_models=False):
        if self.built:
            return

        # We need to build the main task, and potentially a pre-warm task.
        # The default pre-warm task for a model can simply omit the main model,
        # but use the same generator.
        self.task_model = self.build_task()
        if show_models:
            print("******************************************************")
            print("MetaTask (%s) - task model:" % self.name)
            print("******************************************************")
            self.task_model.summary()

        if self.inference_task is not None:
            self.inference_task.build(compile=True)
            if show_models:
                print("******************************************************")
                print("MetaTask (%s) - inference model:" % self.name)
                print("******************************************************")
                self.inference_task.task_model.summary()

        if compile:
            self.task_model.summary()

            ld = self.get_loss_dict()
            lwd = self.get_loss_weight_dict()

            # TODO(b/141715442): experimental_run_tf_function=False is a workaround
            # for b/141715442.
            self.task_model.compile(
                optimizer=self.optimizer,
                loss=self.get_loss_dict(),
                loss_weights=self.get_loss_weight_dict(),
                metrics=self.get_metrics(),
                experimental_run_tf_function=False)

        self.built = True

    def build_task(self, show_models=False):
        training_tasks = [self.main_task] + self.auxillary_tasks

        for task in training_tasks:
            task.build(compile=False)
            if show_models:
                print("******************************************************")
                print("MetaTask (%s) - subtask (%s):" % (self.name, task.name))
                print("******************************************************")
                task.task_model.summary()

            for callback in task.get_callbacks():
                self._add_callback(callback)

            # Clone inputs
            model = task.get_model()
            inputs, names = model_utils.clone_task_inputs(task)
            for n, i in zip(names, inputs):
                self._add_input(n, i)

            # Extract outputs/losses
            task_output_layers = model(inputs)
            if not isinstance(task_output_layers, list):
                task_output_layers = [task_output_layers]
            task_output_names = task.get_output_names()
            task_losses = task.get_loss_dict()
            task_loss_weights = task.get_loss_weight_dict()
            task_metrics = task.get_metrics()

            # Clone the output layers.
            for name, layer in zip(task_output_names, task_output_layers):
                layer = Rename(name=name)(layer)
                loss = task_losses[name]
                loss_weight = task_loss_weights[name]
                metric = task_metrics.get(name, None)

                # TODO - test if this is necessary.
                if isinstance(metric, Pseudometric):
                    metric = Pseudometric(layer)

                self._add_output(name, layer, loss, loss_weight, metric)

        return Model(
            inputs=self.task_inputs, outputs=self.task_outputs, name=self.name)

    def predict(self, x):
        assert self.inference_task is not None, "Must specify an inference task."
        return self.inference_task.predict(x)
