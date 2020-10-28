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
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model

from tensorflow_similarity.api.engine.task import AuxillaryTask
from tensorflow_similarity.utils.model_utils import *
from tensorflow_similarity.layers.rename import Rename

from copy import copy
from PIL import Image

import abc


class RotationModel(abc.ABC):
    @abc.abstractmethod
    def build(self, embedding_shape, embedding_dtype):
        """Build the rotation prediction model for the given embedding.

        Arguments:
            embedding_shape {tuple} -- The shape of the embedding field.
            embedding_dtype {tf.dtype} -- The type of the embedding field.

        Returns:
            Model - - A `Model` which takes as input an embedding, and produces
                a 4-unit softmax, where the four classes are 0, 90, 180, and 270
                degrees.
        """
        return Model()

    def get_config(self):
        return {}


class DefaultRotationModel(RotationModel):
    def build(self, embedding_shape, embedding_dtype):
        i = Input(name="rotation_embedding",
                  dtype=embedding_dtype, shape=embedding_shape)
        o = i
        o = Dense(64, activation="relu")(o)
        o = Dense(8, activation="relu")(o)
        o = Dense(4, activation="softmax")(o)
        return Model(i, o)


def rotate_img(raw_image, rotation_id, reverse=False):
    rotation = rotation_id * 90
    if reverse:
        rotation = rotation * -1

    if raw_image.dtype == np.float:
        raw_image = np.array(raw_image * 255.0, dtype=np.uint8)
        img = Image.fromarray(raw_image)
        img = img.rotate(rotation)
        img = np.array(img, dtype=np.float) / 255.0
    else:
        img = Image.fromarray(raw_image)
        img = img.rotate(rotation)
        img = np.array(img, dtype=np.uint8)

    return img


def rotate_imgs(imgs, rotation_ids, reverse=False):
    num_images = imgs.shape[0]

    rotated_images = []
    for idx, raw_img in enumerate(imgs):
        img = rotate_img(raw_img, rotation_ids[idx], reverse=reverse)
        before = np.sum(raw_img)
        after = np.sum(img)
        rotated_images.append(img)

    rotated_images = np.array(rotated_images)
    return rotated_images


class ImageRotationTask(AuxillaryTask):
    def __init__(self,
                 name,
                 tower_model,
                 image_rotation_model=None,
                 tower_names=[],
                 field_names=[],
                 loss="mse"):
        super(ImageRotationTask, self).__init__(name, tower_model)

        self.name = name
        self.tower_model = tower_model
        self.image_rotation_model = image_rotation_model
        if not image_rotation_model:
            self.image_rotation_model = DefaultRotationModel()
        self.tower_names = tower_names
        self.field_names = field_names
        self.loss = loss

        self.all_input_fields = get_input_names(tower_model)

        self.task_model = None

        self.image_shapes = {}

        input_dictionary = index_inputs(self.tower_model)

        for field in field_names:
            assert field in input_dictionary, "%s is not a field in the model. Known fields: %s" % (
                field, input_dictionary)

    def _input_name(self, tower, field):
        return "%s_%s_%s" % (self.name, tower, field)

    def _source_input_name(self, tower, field):
        return "%s_%s" % (tower, field)

    def _output_name(self, tower, field):
        return "%s_%s_%s_out" % (self.name, tower, field)

    def build_task(self):

        embedding = self.tower_model(self.task_inputs)

        for tower_name in self.tower_names:
            input_names, inputs = clone_model_inputs(
                self.tower_model, prefix="%s_%s_" % (self.name, tower_name))

            for name, inp in zip(input_names, inputs):
                self._add_input(name, inp)

            for field_name in self.field_names:
                input_name = self._input_name(tower_name, field_name)
                output_name = self._output_name(tower_name, field_name)

                input_layer = self.task_input_dict[input_name]
                embedding_shape = layer_shape(embedding)
                image_shape = layer_shape(input_layer)
                self.image_shapes[field_name] = image_shape

                image_rotation_model = self.image_rotation_model.build(
                    embedding_shape, embedding.dtype)

                predicted_rotation = image_rotation_model(embedding)
                output = Rename(name=output_name)(decoded_input)

                self._add_output(output_name, output, self.loss)

        self.task_model = Model(
            inputs=self.task_inputs, outputs=self.task_outputs)

    def update_batch(self, batch):
        """For numeric autoencoders, no manipulation of the inputs is necessary.
        For a string, for instance, some manipulation of the input will may be
        necessary, as there is no practical way to generate a string in
        tensorflow, so an intermediate format (e.g. ordinals) may be necessary.

        Args:
            raw_inputs (dict of string: np.ndarray): Dictionary of feature name
            to raw inputs.
            augmented_inputs (dict of string: np.ndarray): Dictionary of
            feature name to augmented inputs.

        Returns:
            dict of str: np.ndarray: Dictionary of feature name to input for this task.
        """
        feature_dict = batch.values

        inputs = {}
        labels = {}

        for tower_name in self.tower_names:
            for field_name in self.all_input_fields:
                source_name = self._source_input_name(tower_name, field_name)
                input_name = self._input_name(tower_name, field_name)
                output_name = self._output_name(tower_name, field_name)

                raw_images = feature_dict[source_name]
                num_images = raw_images.shape[0]

                rotation_labels = np.random.randint(0, 4, size=(num_images, ))

                rotated_images = rotate_imgs(raw_images, rotation_labels)
                inputs[input_name] = rotated_images
                labels[output_name] = tf.keras.utils.to_categorical(
                    rotation_labels.astype(dtype=np.float32), num_classes=4)

        batch.add_features(self.name, inputs)
        batch.add_labels(self.name, labels)
