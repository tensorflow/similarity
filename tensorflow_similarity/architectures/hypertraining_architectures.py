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

from tensorflow_similarity.architectures.util import maybe_dropout
from tensorflow_similarity.architectures.model_registry import register_model

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from kerastuner.distributions import *


def hyper_democracy(input_shape):
    input = Input(shape=input_shape, name='input')

    OUTPUT_EMBEDDING_SIZE = Choice('embedding_size', [64, 128, 256])
    GRU_SIZE = int(OUTPUT_EMBEDDING_SIZE / 2)
    DROPOUT_STRENGTH = None

    # Dense Network
    x = Flatten()(input)
    dense = Dense(OUTPUT_EMBEDDING_SIZE)(x)
    dense = maybe_dropout(dense, DROPOUT_STRENGTH)

    MIN_FILTER_WIDTH = Choice('min_filter_width', [2, 3, 4])
    NUM_FILTER_WIDTHS = Choice('num_filter_widths', [2, 3, 4])
    MAX_FILTER_WIDTH = MIN_FILTER_WIDTHS + MAX_FILTER_WIDTHS
    MIN_FILTERS = Choice('min_filters', [50, 100, 150, 200])
    FILTER_MULTIPLIER = Choice('filter_count_multipler', [32, 50, 64])
    CONV1D_ACTIVATION = Choice('sepconv1d_activation', ['tanh', 'relu'])

    cnns = []
    # CNN Network
    for size in range(MIN_FILTER_WIDTH, MAX_FILTER_WIDTH):
        num_filters = min(MIN_FILTERS, size * FILTER_MULTIPLIER)
        if CONV1D_ACTIVATION is not None:
            x = SeparableConv1D(
                num_filters, size, activation=CONV1D_ACTIVATION)(input)
        else:
            x = SeparableConv1D(num_filters, size)(input)
        x = BatchNormalization()(x)
        x = maybe_dropout(x, DROPOUT_STRENGTH)
        x = Flatten()(x)
        cnns.append(x)
    cnn = Concatenate()(cnns)
    cnn = Dense(OUTPUT_EMBEDDING_SIZE)(cnn)

    # RNN
    if DROPOUT_STRENGTH:
        rnn = Bidirectional(
            GRU(GRU_SIZE, return_sequences=True,
                dropout=DROPOUT_STRENGTH))(input)
        rnn = Bidirectional(
            GRU(GRU_SIZE, return_sequences=False,
                dropout=DROPOUT_STRENGTH))(rnn)
    else:
        rnn = Bidirectional(GRU(GRU_SIZE, return_sequences=True))(input)
        rnn = Bidirectional(GRU(GRU_SIZE, return_sequences=False))(rnn)

    # Ensemble
    ensemble_add = Add()([rnn, cnn, dense])
    ensemble_mul = Multiply()([rnn, cnn, dense])
    ensemble_max = Maximum()([rnn, cnn, dense])
    ensemble = Concatenate()([ensemble_add, ensemble_mul, ensemble_max])
    ensemble = BatchNormalization()(ensemble)

    output = Dense(OUTPUT_EMBEDDING_SIZE)(ensemble)
    output = BatchNormalization()(output)
    output = maybe_dropout(output, DROPOUT_STRENGTH)
    output = Dense(OUTPUT_EMBEDDING_SIZE)(output)

    model = Model(input, output)
    return model


register_model(hyper_democracy)


def hyper_democracy2(input_shape):
    input = Input(shape=input_shape, name='input')

    OUTPUT_EMBEDDING_SIZE = Choice('embedding_size', [128, 256])
    GRU_SIZE = int(OUTPUT_EMBEDDING_SIZE / 2)
    DROPOUT_STRENGTH = None

    # Dense Network
    x = Flatten()(input)
    dense = Dense(OUTPUT_EMBEDDING_SIZE, activation="selu")(x)
    dense = Dense(OUTPUT_EMBEDDING_SIZE, activation="selu")(dense)
    dense = BatchNormalization()(dense)
    dense = maybe_dropout(dense, DROPOUT_STRENGTH)

    MIN_FILTER_WIDTH = Choice('min_filter_width', [4])
    NUM_FILTER_WIDTHS = Choice('num_filter_widths', [2, 3, 4])
    MAX_FILTER_WIDTH = MIN_FILTER_WIDTH + NUM_FILTER_WIDTHS
    MIN_FILTERS = Choice('min_filters', [100, 150, 200])
    FILTER_MULTIPLIER = Choice('filter_count_multiplier', [32, 50, 64])
    CONV1D_ACTIVATION = Choice('conv1d_activation', ['relu', 'selu'])

    cnns = []
    # CNN Network
    for size in range(MIN_FILTER_WIDTH, MAX_FILTER_WIDTH):
        num_filters = min(MIN_FILTERS, size * FILTER_MULTIPLIER)
        if CONV1D_ACTIVATION is not None:
            x = SeparableConv1D(
                num_filters, size, activation=CONV1D_ACTIVATION)(input)
        else:
            x = SeparableConv1D(num_filters, size)(input)
        x = BatchNormalization()(x)
        x = maybe_dropout(x, DROPOUT_STRENGTH)
        x = Flatten()(x)
        cnns.append(x)
    cnn = Concatenate()(cnns)

    # RNN
    if DROPOUT_STRENGTH:
        rnn = Bidirectional(
            GRU(GRU_SIZE,
                return_sequences=True,
                dropout=DROPOUT_STRENGTH,
                unroll=True))(input)
        rnn = Bidirectional(
            GRU(GRU_SIZE,
                return_sequences=False,
                dropout=DROPOUT_STRENGTH,
                unroll=True))(rnn)
    else:
        rnn = Bidirectional(GRU(GRU_SIZE, return_sequences=True,
                                unroll=True))(input)
        rnn = Bidirectional(
            GRU(GRU_SIZE, return_sequences=False, unroll=True))(rnn)

    # Ensemble
    ensemble = Concatenate()([rnn, cnn, dense])
    ensemble = BatchNormalization()(ensemble)

    output = Dense(OUTPUT_EMBEDDING_SIZE * 2, activation="selu")(ensemble)
    output = BatchNormalization()(output)
    output = Dense(OUTPUT_EMBEDDING_SIZE, activation="selu")(output)
    output = BatchNormalization()(output)
    output = Dense(OUTPUT_EMBEDDING_SIZE, activation="selu")(output)

    model = Model(input, output)
    return model


register_model(hyper_democracy2)
