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

import time

from tensorflow_similarity.api.callbacks.results_callbacks import RenameMetric, ResultWriter
from tensorflow_similarity.api.engine.metrics import Pseudometric, fraction
from tensorflow_similarity.api.engine.task import MainTask
from tensorflow_similarity.api.generators.tuples import HardTripletGenerator, TripletGenerator
from tensorflow_similarity.api.losses.no_loss import NoLoss
from tensorflow_similarity.layers.greater_than import GreaterThan
from tensorflow_similarity.layers.loss_layers import TripletLoss
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils.model_utils import *
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib


class TripletLossTask(MainTask):
    def __init__(self,
                 x,
                 y,
                 tower_model,
                 hard_mining=False,
                 hard_mining_directory="/tmp/",
                 **generator_kwargs):

        if hard_mining:
            self.triplet_generator = HardTripletGenerator(
                x, y, hard_mining_directory=hard_mining_directory, **generator_kwargs)
        else:
            self.triplet_generator = TripletGenerator(x, y, **generator_kwargs)

        super(TripletLossTask,
              self).__init__("triplet_loss", self.triplet_generator,
                             tower_model)
        self.hard_mining = hard_mining
        self.task_model = None
        self.tower_names = ["anchor", "neg", "pos"]

        self._add_callback(
            ResultWriter(self.tower_names,
                         hard_mining_directory=hard_mining_directory,
                         queue_size=2))

    def build_task(self):
        embeddings = []

        devices = get_devices()
        for tower_index, tower in enumerate(self.tower_names):
            prefix = "%s_" % tower
            input_names, inputs = clone_model_inputs(self.tower_model,
                                                     prefix=prefix)
            self.task_inputs.extend(inputs)
            self.task_input_names.extend(input_names)

            idx_input_name = "%s_idx" % tower

            # The index is actually an integer, of course, however, since this
            # is a model output, it needs to be a float32, otherwise the
            # internal loss/weighting code balks at multiplying an int by a
            # float.
            idx_input = Input(name=idx_input_name,
                              shape=(1, ),
                              dtype=tf.float32)
            out_name = "%s_idx_out" % tower
            idx_passthrough = Rename(name=out_name)(idx_input)

            self._add_input(idx_input_name, idx_input)
            self._add_output(
                out_name,
                idx_passthrough,
                loss=NoLoss(),
                metric=Pseudometric(idx_passthrough),
            )
            self.suppressed_metric_prefixes.append(out_name)

            device = devices[tower_index % len(devices)]
            with tf.device(device):
                embedding = self.tower_model(inputs)

            embedding = Rename(name="%s_embedding" % tower)(embedding)
            embeddings.append(embedding)

        # TODO - This was originally necessitated because of a quirk in the way
        # Keras handled multi-gpu models. This should be revisited to see if
        # it's still a required workaround, or if we can simplify this.
        combined_embedding_output = Concatenate(name="combined_embedding",
                                                axis=1)(embeddings)

        triplet_loss = TripletLoss(name="triplets")(combined_embedding_output)
        is_hard = GreaterThan(0.0, name="meta_is_hard")(triplet_loss)
        hard_fraction = Rename(name="hard_tuple")(is_hard)
        self._add_output("triplets", triplet_loss, loss="mae")
        self._add_output("meta_is_hard",
                         is_hard,
                         loss=NoLoss(),
                         metric=Pseudometric(is_hard))
        self._add_output("hard_tuple",
                         hard_fraction,
                         loss=NoLoss(),
                         metric=fraction)
        self.suppressed_metric_prefixes.append("meta_is_hard_pseudometric")

        self.task_model = Model(inputs=self.task_inputs,
                                outputs=self.task_outputs)

    def get_main_batch(self, seq_id):
        batch = self.generator.get_batch(seq_id)
        arbitrary_key = [k for k in batch.values.keys()][0]
        size = len(batch.values[arbitrary_key])

        updates = {
            # Loss values - values computed in the model - we ideally want them to
            # be zero.
            "triplets": np.zeros(size, dtype=np.float32),
            "meta_is_hard": np.zeros(size, dtype=np.float32),
            "hard_tuple": np.zeros(size, dtype=np.float32),
        }
        batch.add_labels("triplet_loss", updates)
        return batch
