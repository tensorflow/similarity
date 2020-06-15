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
from tensorflow_similarity.api.tasks.model_utils import (
    clone_model_input,
    clone_model_inputs,
    compute_size,
    get_input_names,
    index_inputs,
    layer_shape)
from tensorflow_similarity.layers.reduce_mean import ReduceMean
from tensorflow_similarity.layers.reduce_sum import ReduceSum
from tensorflow_similarity.layers.rename import Rename
from tensorflow_similarity.utils.config_utils import register_custom_object


class Decoder(abc.ABC):
    @abc.abstractmethod
    def build(self, embedding_shape, embedding_dtype,
              feature_shape, feature_dtype):
        """Build the decoder for the given embedding and feature Inputs.

        Arguments:
            embedding_shape {tuple} -- The shape of the embedding field.
            embedding_dtype {tf.dtype} -- The type of the embedding field.
            feature_shape {tuple} -- The shape of the feature field.
            feature_dtype {tuple} -- The type of the feature field.

        Returns:
            Model - - A `Model` which computes a reconstruction difference based
                on the provided tensors.
        """
        return Model()

    def get_config(self):
        return {}


class SimpleDecoder(Decoder):
    def reconstruct(self, embedding_shape, embedding_dtype,
                    feature_shape, feature_dtype):
        """Build the decoder for the given embedding and feature Inputs

        Arguments:
            embedding_shape {tuple} -- The shape of the embedding field.
            embedding_dtype {tf.dtype} -- The type of the embedding field.
            feature_shape {tuple} -- The shape of the feature field.
            feature_dtype {tuple} -- The type of the feature field.

        Model - - A `Model` which has a single output, which is the
            reconstruction of the feature based on the embedding.
        """
        return

    def compute_reconstruction_difference(embedding, feature):
        input_size = compute_size(feature)
        embedding_size = compute_size(embedding)
        input_shape = layer_shape(feature)

        decoded_embedding = embedding
        embedding_shape = layer_shape(decoded_embedding)

        if input_size != embedding_size:
            decoded_embedding = Dense(input_size)(embedding)
            embedding_shape = layer_shape(decoded_embedding)

        if not np.array_equal(input_shape, embedding_shape):
            decoded_embedding = Reshape(input_shape)(
                decoded_embedding)

        reconstruction_difference = Subtract()(
            [feature, decoded_embedding])

        return reconstruction_difference

    def build(self, embedding_shape, embedding_dtype,
              feature_shape, feature_dtype):

        embedding = Input(shape=embedding_shape,
                          dtype=embedding_dtype, name="embedding")
        feature = Input(shape=feature_shape,
                        dtype=feature_dtype, name="feature")

        reconstruction_model = self.reconstruct(
            embedding_shape, embedding_dtype, feature_shape, feature_dtype)

        reconstruction = reconstruction_model([embedding, feature])
        difference = self.compute_reconstruction_difference(reconstruction)

        out = difference
        while out.shape.rank > 1:
            out = ReduceSum(axis=-1)(out)

        return Model(
            inputs=[embedding, feature],
            outputs=out,
            name="simple_decoder_wrapper")
