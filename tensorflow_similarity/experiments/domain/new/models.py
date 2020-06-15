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
import tensorflow_tensortext as tensortext
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, AlphaDropout, Convolution1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.models import Model

SELU_ARGS = {
    'activation': "selu",
    'kernel_initializer': "lecun_normal",
    'bias_initializer': 'zeros'
}


def SELU(size):
    return Dense(size, **SELU_ARGS)


def DenseEmbedding(size, **kwargs):
    return Dense(
        size,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        **kwargs)


def dense(input_size=32, embedding_size=32, banks=2, bits=8):
    inputs = Input(shape=(1, ), name='sent_input', dtype='string')
    x = tensortext.layers.CharSeq(
        sequence_length=input_size, banks=banks, bits_per_bank=bits)(inputs)
    mem = Flatten()(x)
    mem = SELU(128)(mem)
    mem = DenseEmbedding(embedding_size)(mem)
    m = Model(inputs, mem)
    return m


def kim_char_cnn_ngrams(input_size=32,
                        max_grams=3,
                        alphabet_size=69,
                        tensortext_decoder_layers=1,
                        tensortext_decoder_layer_size=64,
                        conv_layers=[[256, 10], [256, 7], [256, 5], [256, 3]],
                        fully_connected_layers=[512, 512, 256],
                        dropout_p=.1,
                        embedding_size=32,
                        banks=2,
                        bits=8,
                        optimizer='adam',
                        loss='categorical_crossentropy'):

    # Input layer
    inputs = Input(shape=(1, ), name='example', dtype='string')
    # Embedding layers

    x = tensortext.layers.CharSeq(
        sequence_length=input_size, banks=banks, bits_per_bank=bits)(inputs)

    mem = Flatten()(x)
    mem = SELU(256)(mem)
    mem = SELU(128)(mem)

    ngrams_sequences = []
    ngrams_sequences.append(x)

    for n in range(2, max_grams + 1):
        ngram_sequence = tensortext.layers.SequenceToNGrams(N=n)(x)
        ngrams_sequences.append(ngram_sequence)

    outputs = []
    outputs.append(mem)
    for n, ngram_sequence in enumerate(ngrams_sequences):
        kim_outputs = []
        for num_filters, filter_width in conv_layers:
            conv = Convolution1D(
                filters=num_filters,
                kernel_size=filter_width,
                activation='tanh',
                name='Conv1D_{}_{}_{}gram'.format(num_filters, filter_width,
                                                  n + 1))(ngram_sequence)
            pool = GlobalMaxPooling1D(
                name='MaxPoolingOverTime_{}_{}_{}gram'.format(
                    num_filters, filter_width, n + 1))(conv)
            kim_outputs.append(pool)
        outputs.append(SELU(512)(Concatenate()(kim_outputs)))

    # Fully connected layers
    fc = Concatenate()(outputs)
    for layer_size in fully_connected_layers:
        fc = SELU(layer_size)(fc)
    fc = AlphaDropout(.1)(fc)

    # Output layer
    predictions = DenseEmbedding(embedding_size, name="embedding")(fc)
    # Build and compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
    model.summary()
    return model


def ensemble_model():
    input_size = 32
    alphabet_size = 69
    tensortext_decoder_layers = 1
    tensortext_decoder_layer_size = 64
    embedding_size = 300
    conv_layers = [[256, 10], [256, 7], [256, 5], [256, 3]]
    fully_connected_layers = [512]
    final_layer_size = 128
    dropout_p = .1
    banks = 16
    bits = 8
    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    input_seq = Input(shape=(1, ), name='example', dtype=tf.string)

    x = tensortext.layers.CharSeq(
        sequence_length=input_size, banks=8, bits_per_bank=8,
        input_bits=8)(input_seq)

    submodels = []

    dense = SELU(1024)(Flatten()(x))
    dense = SELU(512)(dense)
    dense = SELU(512)(dense)
    submodels.append(dense)

    # Convolution layers
    conv_input = Bidirectional(x)
    convolution_output = []
    for num_filters, filter_width in conv_layers:
        conv = Convolution1D(
            filters=num_filters,
            kernel_size=filter_width,
            activation='tanh',
            name='Conv1D_{}_{}'.format(num_filters, filter_width))(conv_input)
        pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(
            num_filters, filter_width))(conv)
        convolution_output.append(pool)
    c = Concatenate()(convolution_output)
    submodels.append(c)

    ensemble = Concatenate(axis=1)(submodels)
    ensemble = SELU(512)(ensemble)
    ensemble = SELU(256)(ensemble)
    ensemble = AlphaDropout(dropout_p)(ensemble)
    out = DenseEmbedding(256)(ensemble)

    model = Model(inputs=input_seq, outputs=out)
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model
