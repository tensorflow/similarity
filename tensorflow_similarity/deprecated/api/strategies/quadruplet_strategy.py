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

import json
import multiprocessing
import tempfile
import warnings

import tensorflow as tf
from tensorflow_similarity.api.engine.augmentation import Augmentation
from tensorflow_similarity.api.engine.database import Database
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.simhash import SimHashInterface, SimHashRegistry
from tensorflow_similarity.api.engine.similarity_model import SimilarityModel
from tensorflow_similarity.api.engine.task import MetaTask, Task
from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask
from tensorflow_similarity.api.tasks.inference import InferenceTask
from tensorflow_similarity.api.tasks.prewarm_task import create_prewarming_task
from tensorflow_similarity.api.tasks.quadruplet_loss_task import QuadrupletLossTask
from tensorflow_similarity.api.tasks.stable_quadruplet_loss_task import \
    StableQuadrupletLossTask
from tensorflow_similarity.callbacks.hard_mining import ResultWriter
from tensorflow_similarity.utils.config_utils import register_custom_object
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import deserialize_keras_object as deserialize
from tensorflow.keras.utils import serialize_keras_object as serialize


class QuadrupletLossStrategy(SimilarityModel):
    def __init__(self,
                 tower_model=None,
                 auxillary_tasks=[],
                 strategy="quadruplet_loss",
                 preprocessing=None,
                 augmentation=None,
                 optimizer=None,
                 name="quadruplet_loss",
                 **generator_config):

        super(QuadrupletLossStrategy, self).__init__(
            tower_model=tower_model,
            auxillary_tasks=auxillary_tasks,
            strategy="quadruplet_loss",
            preprocessing=preprocessing,
            augmentation=augmentation,
            optimizer=optimizer,
            name=name,
            towers=["anchor", "neg1", "neg2", "pos"],
            **generator_config)

    def _build_training_task(self, x, y):
        main_task = QuadrupletLossTask(x,
                                       y,
                                       self.tower_model,
                                       hard_mining=False,
                                       **self.generator_config)

        aux_tasks = self._build_auxillary_tasks(x, y)

        meta_task = MetaTask("quadruplet_loss",
                             self.tower_model,
                             main_task,
                             auxillary_tasks=aux_tasks,
                             optimizer=self.optimizer)

        meta_task.build(compile=True)
        return meta_task


# To allow SimHash(...) to construct this strategy.
SimHashRegistry.register("quadruplet_loss", QuadrupletLossStrategy)
# To allow keras's model deserialization to load the model.
register_custom_object("QuadrupletLossStrategy", QuadrupletLossStrategy)


class HardQuadrupletLossStrategy(SimilarityModel):
    def __init__(self, tower_model, hard_mining_directory=None, **kwargs):
        self.hard_mining_directory = hard_mining_directory
        if not hard_mining_directory:
            self.hard_mining_directory = tempfile.mkdtemp()

        super(HardQuadrupletLossStrategy,
              self).__init__(tower_model,
                             name="hard_quadruplet_loss",
                             **kwargs)

    def _build_training_task(self, x, y):
        main_task = QuadrupletLossTask(x,
                                       y,
                                       self.tower_model,
                                       hard_mining=True,
                                       **self.generator_config)

        aux_tasks = self._build_auxillary_tasks(x, y)

        meta_task = MetaTask("hard_quadruplet_loss",
                             self.tower_model,
                             main_task,
                             auxillary_tasks=aux_tasks,
                             optimizer=self.optimizer)

        meta_task.build(compile=True)

        return meta_task


SimHashRegistry.register("hard_quadruplet_loss", HardQuadrupletLossStrategy)
register_custom_object("HardQuadrupletLossStrategy",
                       HardQuadrupletLossStrategy)


class StableQuadrupletLossStrategy(SimilarityModel):
    def __init__(self, tower_model, **kwargs):

        super(StableQuadrupletLossStrategy,
              self).__init__(tower_model,
                             name="stable_quadruplet_loss",
                             **kwargs)

    def _build_training_task(self, x, y):
        main_task = StableQuadrupletLossTask(
            x,
            y,
            self.tower_model,
            hard_mining=False,
            **self.generator_config)

        aux_tasks = self._build_auxillary_tasks(x, y)

        meta_task = MetaTask("stable_quadruplet_loss",
                             self.tower_model,
                             main_task,
                             auxillary_tasks=aux_tasks,
                             optimizer=self.optimizer)

        meta_task.build(compile=True)

        return meta_task


SimHashRegistry.register("stable_quadruplet_loss",
                         StableQuadrupletLossStrategy)
register_custom_object("StableQuadrupletLossStrategy",
                       StableQuadrupletLossStrategy)


class StableHardQuadrupletLossStrategy(SimilarityModel):
    def __init__(self, tower_model, hard_mining_directory=None, **kwargs):
        self.hard_mining_directory = hard_mining_directory
        if not hard_mining_directory:
            self.hard_mining_directory = tempfile.mkdtemp()

        super(StableHardQuadrupletLossStrategy,
              self).__init__(tower_model,
                             name="stable_hard_quadruplet_loss",
                             **kwargs)

    def _build_training_task(self, x, y):
        main_task = StableQuadrupletLossTask(
            x,
            y,
            self.tower_model,
            hard_mining=True,
            hard_mining_directory=self.hard_mining_directory,
            **self.generator_config)

        aux_tasks = self._build_auxillary_tasks(x, y)

        meta_task = MetaTask("stable_hard_quadruplet_loss",
                             self.tower_model,
                             main_task,
                             auxillary_tasks=aux_tasks,
                             optimizer=self.optimizer)

        meta_task.build(compile=True)

        return meta_task


SimHashRegistry.register("stable_hard_quadruplet_loss",
                         StableHardQuadrupletLossStrategy)
register_custom_object("StableHardQuadrupletLossStrategy",
                       StableHardQuadrupletLossStrategy)
