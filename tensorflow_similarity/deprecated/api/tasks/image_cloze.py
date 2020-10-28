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

from copy import copy

from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras.models import Model, model_from_config
from tensorflow.keras.utils import deserialize_keras_object as deserialize
from tensorflow.keras.utils import serialize_keras_object as serialize

from tensorflow_similarity.api.engine.decoder import Decoder, SimpleDecoder
from tensorflow_similarity.api.engine.task import AuxillaryTask
from tensorflow_similarity.api.tasks import AutoencoderTask
from tensorflow_similarity.utils.model_utils import *
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils.config_utils import register_custom_object


def blank_out_patch(image, patch, position):
    shape = image.shape
    damaged = copy(image)

    x, y = position

    x_end = x + patch[0]
    y_end = y + patch[1]

    zeros = np.zeros((patch[0], patch[1], shape[2]), dtype=np.float32)

    damaged[x:x_end, y:y_end, :] = zeros
    return damaged


def blank_out_patches(imgs, patch):
    shape = imgs.shape

    if len(shape) == 4:
        _, w, h, _ = shape
    else:
        _, w, h = shape
        channels = 0

    out = []
    for img in imgs:
        x = np.random.randint(0, w - patch[0])
        y = np.random.randint(0, h - patch[1])
        damaged = blank_out_patch(img, patch, (x, y))
        out.append(damaged)

    return out


class ExampleReconstructionModel(SimpleDecoder):
    # TODO - change this to a conv-net
    def build_reconstruction_model(self):
        embedding = self.create_embedding_input()
        o = Dense(1024)(embedding)
        o = Dense(2048)(o)
        o = self.feature_shaped_dense(o)
        m = Model(embedding, o, name="reconstruction")
        m.compile(loss="mae", optimizer="adam")
        return m


register_custom_object("ExampleReconstructionModel",
                       ExampleReconstructionModel)


class ImageClozeTask(AutoencoderTask):
    def __init__(self,
                 name,
                 tower_model,
                 decoder,
                 tower_names,
                 field_names,
                 loss="mae",
                 loss_weight=1.0,
                 input_preprocessing=None,
                 target_preprocessing=None,
                 input_feature_type="augmented",
                 target_feature_type="augmented",
                 patch_size=(4, 4)):
        self.patch_size = patch_size
        super(ImageClozeTask,
              self).__init__(name,
                             tower_model,
                             decoder,
                             tower_names,
                             field_names,
                             loss=loss,
                             loss_weight=loss_weight,
                             input_preprocessing=input_preprocessing,
                             target_preprocessing=target_preprocessing,
                             input_feature_type=input_feature_type,
                             target_feature_type=target_feature_type)

    def update_batch(self, batch):
        """For numeric autoencoders, no manipulation of the inputs is necessary.
        For a string, for instance, some manipulation of the input will may be
        necessary, as there is no practical way to generate a string in
        tensorflow, so an intermediate format(e.g. ordinals) may be necessary.

        Args:
            batch (Batch) -- The input batch, so far.

        Returns:
            Batch: the input Batch, including features/labels for this task.
        """
        super(ImageClozeTask, self).update_batch(batch)

        updates = {}

        for tower_name in self.tower_names:
            for field_name in self.all_input_fields:
                input_name = self._input_name(tower_name, field_name)

                # Note that we're modifying the field already set by the
                # Autoencoder.update_batch() call, so it will always be the
                # "augmented" feature here.
                images = batch.get(input_name, val_type="augmented")
                images = blank_out_patches(images, self.patch_size)
                updates[input_name] = np.array(images)

        batch.update_features(self.name, updates)

    def get_config(self):
        config = super(ImageClozeTask, self).get_config()
        config["patch_size"] = self.patch_size
        return config


register_custom_object("ImageClozeTask", ImageClozeTask)
