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

import abc

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Subtract
from tensorflow.keras.models import Model, model_from_config
from tensorflow.keras.utils import deserialize_keras_object as deserialize
from tensorflow.keras.utils import serialize_keras_object as serialize

from tensorflow_similarity.api.engine.parameterized_model import ParameterizedModel
from tensorflow_similarity.api.engine.preprocessing import Preprocessing
from tensorflow_similarity.api.engine.task import AuxillaryTask, Task
from tensorflow_similarity.layers.reduce_mean import ReduceMean
from tensorflow_similarity.layers.reduce_sum import ReduceSum
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils.config_utils import register_custom_object
from tensorflow_similarity.utils.model_utils import (
    clone_model_input,
    clone_model_inputs,
    compute_size,
    get_input_names,
    index_inputs,
    layer_shape)


class Decoder(abc.ABC):
    def __init__(self, name="decoder"):
        self.name = name

        self.embedding_shape = None
        self.embedding_dtype = None
        self.embedding_input = None
        self.embedding_size = None

        self.feature_shape = None
        self.feature_dtype = None
        self.feature_input = None
        self.feature_size = None

    def set_parameters(self, embedding_shape, embedding_dtype, feature_shape,
                       feature_dtype):
        self.embedding_shape = embedding_shape
        self.embedding_dtype = embedding_dtype
        self.embedding_size = compute_size(self.embedding_shape)

        self.feature_shape = feature_shape
        self.feature_dtype = feature_dtype
        self.feature_size = compute_size(feature_shape)

    @abc.abstractmethod
    def build_decoder_model(self):
        """Build the decoder for the parameters set on this object.

        Returns:
            Model - - A `Model` with two inputs, the embedding and the feature,
                and one output - the reconstruction loss.
        """
        return

    def create_embedding_input(self):
        return Input(shape=self.embedding_shape,
                     dtype=self.embedding_dtype,
                     name="%s_embedding" % self.name)

    def create_feature_input(self):
        return Input(shape=self.feature_shape,
                     dtype=self.feature_dtype,
                     name="%s_feature" % self.name)

    def build(self, embedding_shape, embedding_dtype, feature_shape,
              feature_dtype):
        """Build the decoder for the given embedding and feature input
        descriptions.

        Arguments:
            embedding_shape {tuple} -- The shape of the embedding field.
            embedding_dtype {tf.dtype} -- The type of the embedding field.
            feature_shape {tuple} -- The shape of the feature field.
            feature_dtype {tuple} -- The type of the feature field.

        Returns:
            Model - - A `Model` with two inputs, the embedding and the feature,
                and one output - the reconstruction loss.
        """
        self.set_parameters(embedding_shape, embedding_dtype, feature_shape,
                            feature_dtype)
        return self.build_decoder_model()

    def feature_sized_dense(self, layer, name="feature_sized_dense", **kwargs):
        return Dense(self.feature_size, name=name, **kwargs)(layer)

    def reshape_to_feature_shape(self, layer):
        return Reshape(self.feature_shape,
                       name="reshape_to_feature_shape")(layer)

    def feature_shaped_dense(self, layer, **kwargs):
        o = self.feature_sized_dense(layer)
        return self.reshape_to_feature_shape(o)

    def get_config(self):
        return {
            'name': self.name
        }


class SimpleDecoder(Decoder):
    @abc.abstractmethod
    def build_reconstruction_model(self):
        """Build the decoder for the parameters set on this Decoder.

        Model - - A `Model` which has a single input - the embedding value,
            and a single output, which is the reconstruction of the feature
            based on the embedding.
        """
        return

    def compute_reconstruction_difference(self, reconstruction, feature):
        input_size = compute_size(feature)
        embedding_size = compute_size(reconstruction)
        input_shape = layer_shape(feature)

        decoded_embedding = reconstruction
        embedding_shape = layer_shape(decoded_embedding)

        if input_size != embedding_size:
            decoded_embedding = Dense(input_size)(reconstruction)
            embedding_shape = layer_shape(decoded_embedding)

        if not np.array_equal(input_shape, embedding_shape):
            decoded_embedding = Reshape(input_shape)(decoded_embedding)

        reconstruction_difference = Subtract()([feature, decoded_embedding])

        return reconstruction_difference

    def build_decoder_model(self):
        embedding = self.create_embedding_input()
        feature = self.create_feature_input()

        reconstruction_model = self.build_reconstruction_model()
        reconstruction = reconstruction_model([embedding, feature])
        difference = self.compute_reconstruction_difference(
            reconstruction, feature)

        model = Model(inputs=[embedding, feature],
                      outputs=difference,
                      name="simple_decoder_wrapper")
        return model
