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

import multiprocessing
import tempfile
import warnings

from tensorflow.keras.optimizers import Adam

from tensorflow_similarity.api.engine.database import Database
from tensorflow_similarity.api.engine.simhash import SimHashInterface, SimHashRegistry
from tensorflow_similarity.api.engine.similarity_model import SimilarityModel
from tensorflow_similarity.api.engine.task import MetaTask

from tensorflow_similarity.api.tasks.autoencoder import AutoencoderTask
from tensorflow_similarity.api.tasks.inference import InferenceTask
from tensorflow_similarity.api.tasks.triplet_loss_task import TripletLossTask
from tensorflow_similarity.api.tasks.prewarm_task import create_prewarming_task

from tensorflow_similarity.callbacks.hard_mining import ResultWriter

from tensorflow_similarity.utils.config_utils import register_custom_object


class TripletLossStrategy(SimilarityModel):
    def __init__(
            self,
            tower_model,
            auxillary_tasks=[],
            strategy="triplet_loss",
            preprocessing=None,
            augmentation=None,
            optimizer=None,
            name="triplet_loss",
            **generator_config):

        super(TripletLossStrategy, self).__init__(
            tower_model=tower_model,
            auxillary_tasks=auxillary_tasks,
            strategy=strategy,
            preprocessing=preprocessing,
            augmentation=augmentation,
            optimizer=optimizer,
            name=name,
            towers=["anchor", "neg", "pos"],
            **generator_config)

    def _build_training_task(self, x, y):
        main_task = TripletLossTask(
            x,
            y,
            self.tower_model,
            hard_mining=False,
            **self.generator_config)

        aux_tasks = self._build_auxillary_tasks(x, y)

        meta_task = MetaTask(
            "triplet_loss",
            self.tower_model,
            main_task,
            auxillary_tasks=aux_tasks,
            optimizer=self.optimizer)

        meta_task.build(compile=True)
        return meta_task


# To allow SimHash(...) to construct this strategy.
SimHashRegistry.register("triplet_loss", TripletLossStrategy)
# To allow keras's model deserialization to load the model.
register_custom_object("TripletLossStrategy", TripletLossStrategy)


class HardTripletLossStrategy(SimilarityModel):
    def __init__(self,
                 tower_model,
                 auxillary_tasks=[],
                 strategy="hard_triplet_loss",
                 preprocessing=None,
                 augmentation=None,
                 optimizer=None,
                 name="triplet_loss",
                 hard_mining_directory=None,
                 **generator_config):

        super(HardTripletLossStrategy, self).__init__(
            tower_model=tower_model,
            auxillary_tasks=auxillary_tasks,
            strategy=strategy,
            preprocessing=preprocessing,
            augmentation=augmentation,
            optimizer=optimizer,
            name=name,
            hard_mining=True,
            towers=["anchor", "neg", "pos"],
            hard_mining_directory=hard_mining_directory,
            **generator_config)

    def _build_training_task(self, x, y):
        main_task = TripletLossTask(
            x,
            y,
            self.tower_model,
            hard_mining=True,
            hard_mining_directory=self.hard_mining_directory,
            **self.generator_config)

        aux_tasks = self._build_auxillary_tasks(x, y)

        meta_task = MetaTask(
            "hard_triplet_loss",
            self.tower_model,
            main_task,
            auxillary_tasks=aux_tasks,
            optimizer=self.optimizer)

        meta_task.build(compile=True)
        return meta_task


SimHashRegistry.register("hard_triplet_loss", HardTripletLossStrategy)
register_custom_object("HardTripletLossStrategy", HardTripletLossStrategy)
