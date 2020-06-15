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

from tensorflow_similarity.architectures import util
from tensorflow.keras import backend as K

from tensorflow_similarity.architectures.elienet_v2 import elienet_v2, elienet_v2_decoder
from tensorflow_similarity.architectures.model_registry import register_model
import tensorflow as tf
import numpy as np
try:
    from minipouce.layers import *
except:
    pass

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

SELU_ARGS = {
    'activation': "selu",
    'kernel_initializer': "lecun_normal",
    'bias_initializer': 'zeros'
}


def SELU(size):
    return Dense(size, **SELU_ARGS)


def DenseEmbedding(size):
    return Dense(
        size,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))


def flattener(shape):
    """Flattener model. Can be used in conjunction with minipouce as a baseline
    model."""
    base_input = Input(shape=shape["example"], name='example')
    output = Flatten()(base_input)
    model = Model(base_input, output)
    return model


register_model(flattener)


def passthrough(shape):
    """Passthrough model.  Output == input. Can be used as a
    baseline model for low dimensional inputs."""
    base_input = Input(shape=shape["example"], name='example')
    output = base_input
    model = Model(base_input, output)
    return model


register_model(passthrough)


def one_layer(shape):
    base_input = Input(shape=shape["example"], name='example')
    x = Flatten()(base_input)
    output = Dense(512)(x)
    model = Model(base_input, output)
    return model


register_model(one_layer)


def two_layer(shape):
    base_input = Input(shape=shape["example"], name='example')
    x = Flatten()(base_input)
    x = Dense(512)(x)
    output = Dense(256)(x)
    model = Model(base_input, output)
    return model


register_model(two_layer)


def iris(shape):
    base_input = Input(shape=shape["example"], name='example')
    x = Flatten()(base_input)
    x = SELU(16)(x)
    x = SELU(16)(x)
    x = SELU(16)(x)
    output = Dense(2, activation="sigmoid")(x)
    model = Model(base_input, output)
    return model


register_model(iris)


def two_layer_dual_minipouce(shape):
    input1 = Input(shape=shape["example"], name='example')
    input2 = Input(shape=shape["example_vec"], name='example_vec')

    x = Flatten()(input1)
    x = Dense(128)(x)
    y = Dense(128)(input2)
    x = Concatenate()([x, y])
    output = Dense(64)(x)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


register_model(two_layer_dual_minipouce)


def dense_dual_minipouce(shape):

    input1 = Input(shape=shape["example"], name='example')
    input2 = Input(shape=shape["example_vec"], name='example_vec')

    x = Bidirectional(GRU(32, unroll=True, return_sequences=True))(input1)
    x = Bidirectional(GRU(32, unroll=True, return_sequences=False))(x)
    x = Dense(128, activation="relu")(x)

    y = Dense(128, activation="relu")(input2)
    y = Dense(128, activation="relu")(y)
    y = Dense(128, activation="relu")(y)

    x = Concatenate()([x, y])
    output = Dense(128, activation="selu")(x)
    model = Model(inputs=[input1, input2], outputs=output)

    return model


register_model(dense_dual_minipouce)


def wide_and_gru(shape):
    base_input = Input(shape=shape['example'], name='input')
    x = base_input
    y = Flatten()(x)
    y = Dense(128, activation='relu')(y)
    x = Bidirectional(GRU(64, return_sequences=True, unroll=True))(x)
    x = Bidirectional(GRU(64, unroll=True))(x)
    x = Dense(64, activation='relu')(x)
    x = Concatenate()([x, y])
    x = Dense(64, activation='relu')(x)
    output = Dense(64, activation='sigmoid')(x)
    model = Model(base_input, output)
    return model


register_model(wide_and_gru)


def decoding_gru(shape):
    base_input = Input(shape=shape['example'], name='input')
    x = base_input
    y = Flatten()(x)
    y = Dense(128, activation='relu')(y)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Bidirectional(GRU(64, return_sequences=True, unroll=True))(x)
    x = Bidirectional(GRU(64, unroll=True))(x)
    x = Dense(64, activation='relu')(x)
    x = Concatenate()([x, y])
    x = Dense(64, activation='relu')(x)
    output = Dense(64, activation='sigmoid')(x)
    model = Model(base_input, output)
    return model


register_model(decoding_gru)


def hierarchical_dense(shape):
    base_input = Input(shape=shape, name='input')
    x = base_input
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(64, activation='sigmoid')(x)
    model = Model(base_input, output)
    return model


register_model(hierarchical_dense)


def wide_and_deep(shape):
    base_input = Input(shape=shape['example'], name='input')
    x = base_input

    x = Flatten()(x)
    y = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Concatenate()([x, y])
    x = Dense(64, activation='relu')(x)
    output = Dense(64, activation='sigmoid')(x)
    model = Model(base_input, output)
    return model


register_model(wide_and_deep)


def dense(shape):
    base_input = Input(shape=shape['example'], name='example')
    x = base_input
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(256, activation='sigmoid')(x)
    model = Model(base_input, output)
    return model


register_model(dense)


class MyHighway(Layer):
    """Densely connected highway network."""

    def __init__(self,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):

        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, )
        super(MyHighway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='W_carry')
        if self.bias:
            self.b = self.add_weight((input_dim, ),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim, ),
                                           initializer='one',
                                           name='b_carry')
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = activations.sigmoid(y)
        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {
            'init':
            initializers.serialize(self.init),
            'activation':
            activations.serialize(self.activation),
            'W_regularizer':
            regularizers.serialize(self.W_regularizer),
            'b_regularizer':
            regularizers.serialize(self.b_regularizer),
            'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
            'W_constraint':
            constraints.serialize(self.W_constraint),
            'b_constraint':
            constraints.serialize(self.b_constraint),
            'bias':
            self.bias,
            'input_dim':
            self.input_dim
        }
        base_config = super(MyHighway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def highway(input_shape):
    EMBEDDING_DIM = 16
    input = Input(shape=input_shape, name='input')

    x = Embedding(20000, EMBEDDING_DIM)(input)
    embed_model = Model(input, x)
    convs = []
    first_conv = None
    max_pool = None
    # Filter sizes are different from paper.
    for kernel_size in [1, 2, 3, 4, 5, 6, 7]:
        num_filters = min(200, kernel_size * 50)
        # Paper uses tanh for conv layer.
        conv = Conv2D(num_filters, [1, kernel_size], activation='relu')(x)
        # Paper is not using batch norm
        pooled = Lambda(lambda x: K.max(x, axis=[2]))(conv)
        if not first_conv:
            first_conv = Model(input, conv)
            max_pool = Model(input, pooled)
        convs.append(pooled)

    x = Concatenate()(convs)
    for i in range(2):
        x = TimeDistributed(MyHighway(activation='relu'))(x)

    lstm_input_model = Model(input=input, output=[x])
    x = LSTM(
        650,
        activation='tanh',
        return_sequences=True,
        input_shape=(None, None))(x)
    x = LSTM(650, dropout=0.5)(x)
    x = Dense(64, activation='relu')(x)

    model = Model(input, x)
    return model


register_model(highway)


def tcn2(input_shape):
    input = Input(shape=input_shape, name='input')

    td = TimeDistributed(Dense(64, activation='relu'))(input)
    td = TimeDistributed(Dense(64, activation='relu'))(td)
    td = TimeDistributed(Dense(64, activation='relu'))(td)
    td = Flatten()(td)

    cnns = []
    for size in [3, 5, 7]:
        x = SeparableConv1D(32, size, activation='relu')(input)
        x = Flatten()(x)
        cnns.append(x)

    cnns.append(td)

    x = Concatenate()(cnns)
    x = RepeatVector(30)(x)
    x = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.2, unroll=True))(x)
    x = Bidirectional(
        GRU(128, return_sequences=False, dropout=0.2, unroll=True))(x)

    model = Model(input, x)
    return model


register_model(tcn2)


def tcn3(input_shape):
    input = Input(shape=input_shape, name='input')

    cnns = []
    for size in range(2, 7):
        num_filters = size * 32  # Experiment: min(150..., size*50)
        # no activation was worse
        x = separable_conv_1d(num_filters, size, activation='tanh')(input)

        # Experiment with multiple layers?
        # x = separable_conv_1d(num_filters, size, activation='relu')(input)
        # Maybe add a dense
        # x = Dense(100 * size,activation='relu')(x) # Probably no gain
        x = Flatten()(x)

        cnns.append(x)

    x = Concatenate()(cnns)

    x = RepeatVector(10)(x)
    x = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.2, unroll=True))(x)
    x = Bidirectional(
        GRU(128, return_sequences=False, dropout=0.2, unroll=True))(x)
    x = Dense(256)(x)

    model = Model(input, x)
    model.compile(optimizer="adam", loss='mse')
    return model


register_model(tcn3)


def democracy(input_shape):
    input = Input(shape=input_shape['example'], name='example')

    # Dense Network
    x = Flatten()(input)
    dense = Dense(512)(x)
    dense = Dropout(.2)(dense)

    cnns = []
    # CNN Network
    for size in [3]:
        num_filters = min(150, size * 50)
        x = SeparableConv1D(
            num_filters, size,
            activation='tanh')(input)  # no activation was worse
        x = Dropout(.2)(x)
        x = Flatten()(x)
        cnns.append(x)

    #    cnn = Concatenate()(cnns)
    cnn = cnns[0]
    cnn = Dense(512)(cnn)

    # RNN
    rnn = Bidirectional(
        GRU(256, return_sequences=True, dropout=.2, unroll=True))(input)
    rnn = Bidirectional(
        GRU(256, return_sequences=False, dropout=.2, unroll=True))(rnn)

    # Ensemble
    ensemble_add = Add()([rnn, cnn, dense])
    ensemble_mul = Multiply()([rnn, cnn, dense])
    ensemble_max = Maximum()([rnn, cnn, dense])
    ensemble = Concatenate()([ensemble_add, ensemble_mul, ensemble_max])

    output = Dense(256)(ensemble)
    output = Dropout(.2)(output)
    output = Dense(256)(output)

    model = Model(input, output)
    model.compile(optimizer="adam", loss='mse')
    return model


register_model(democracy)


def _mp(input_shape, args):
    def Choice(name, things):
        """Analog of kerastuner.distributions.Choice which fixes the
      parameter based on kwargs"""
        return args[name]

    inp = Input(shape=input_shape, name='input', dtype="string")
    pp_inp = Seq2Seq(vocab_size=10000)(inp)

    OUTPUT_EMBEDDING_SIZE = Choice('embedding_size', [64, 128, 256])
    GRU_SIZE = int(OUTPUT_EMBEDDING_SIZE / 2)
    DROPOUT_STRENGTH = None

    # Dense Network
    dense = Flatten()(pp_inp)
    dense = Dense(OUTPUT_EMBEDDING_SIZE)(dense)
    dense = maybe_dropout(dense, DROPOUT_STRENGTH)

    MIN_FILTER_WIDTH = Choice('min_filter_width', [2, 3, 4])
    NUM_FILTER_WIDTHS = Choice('num_filter_widths', [2, 3, 4])
    MAX_FILTER_WIDTH = MIN_FILTER_WIDTH + NUM_FILTER_WIDTHS
    MIN_FILTERS = Choice('min_filters', [100, 150, 200])
    FILTER_MULTIPLIER = Choice('filter_count_multiplier', [32, 50, 64])
    CONV1D_ACTIVATION = Choice('conv1d_activation', ['tanh', 'relu'])

    cnns = []
    # CNN Network
    for size in range(MIN_FILTER_WIDTH, MAX_FILTER_WIDTH):
        num_filters = min(MIN_FILTERS, size * FILTER_MULTIPLIER)
        if CONV1D_ACTIVATION is not None:
            x = SeparableConv1D(
                num_filters, size, activation=CONV1D_ACTIVATION)(pp_inp)
        else:
            x = SeparableConv1D(num_filters, size)(pp_inp)
        x = maybe_dropout(x, DROPOUT_STRENGTH)
        x = Flatten()(x)
        cnns.append(x)
    cnn = Concatenate()(cnns)
    cnn = Dense(OUTPUT_EMBEDDING_SIZE)(cnn)

    # RNN
    if DROPOUT_STRENGTH:
        rnn = Bidirectional(
            GRU(GRU_SIZE,
                return_sequences=True,
                dropout=DROPOUT_STRENGTH,
                unroll=True))(pp_inp)
        rnn = Bidirectional(
            GRU(GRU_SIZE,
                return_sequences=False,
                dropout=DROPOUT_STRENGTH,
                unroll=True))(rnn)
    else:
        rnn = Bidirectional(GRU(GRU_SIZE, return_sequences=True,
                                unroll=True))(pp_inp)
        rnn = Bidirectional(
            GRU(GRU_SIZE, return_sequences=False, unroll=True))(rnn)

    # Ensemble
    ensemble_add = Add()([rnn, cnn, dense])
    ensemble_mul = Multiply()([rnn, cnn, dense])
    ensemble_max = Maximum()([rnn, cnn, dense])
    ensemble = Concatenate()([ensemble_add, ensemble_mul, ensemble_max])

    output = Dense(OUTPUT_EMBEDDING_SIZE)(ensemble)
    output = maybe_dropout(output, DROPOUT_STRENGTH)
    output = Dense(OUTPUT_EMBEDDING_SIZE, activation='sigmoid')(output)

    model = Model(inp, output)
    model.compile(optimizer="adam", loss='mse')
    return model


def mp(input_shape):
    return _mp(
        input_shape, {
            "embedding_size": 256,
            "min_filter_width": 5,
            "num_filter_widths": 2,
            "filter_count_multiplier": 32,
            "min_filters": 200,
            "conv1d_activation": "relu",
        })


register_model(mp)


def _dual_mp(input_shape, args):
    def Choice(name, things):
        """Analog of kerastuner.distributions.Choice which fixes the
      parameter based on kwargs"""
        return args[name]

    OUTPUT_EMBEDDING_SIZE = Choice('embedding_size', [64, 128, 256])
    GRU_SIZE = int(OUTPUT_EMBEDDING_SIZE / 2)
    DROPOUT_STRENGTH = None
    MIN_FILTER_WIDTH = Choice('min_filter_width', [2, 3, 4])
    NUM_FILTER_WIDTHS = Choice('num_filter_widths', [2, 3, 4])
    MAX_FILTER_WIDTH = MIN_FILTER_WIDTH + NUM_FILTER_WIDTHS
    MIN_FILTERS = Choice('min_filters', [100, 150, 200])
    FILTER_MULTIPLIER = Choice('filter_count_multiplier', [32, 50, 64])
    CONV1D_ACTIVATION = Choice('conv1d_activation', ['tanh', 'relu'])

    input_seq = Input(shape=input_shape["example"], name='example')
    input_vec = Input(shape=input_shape["example_vec"], name='example_vec')

    dense = Dense(OUTPUT_EMBEDDING_SIZE / 2, activation='relu')(input_vec)
    dense = maybe_dropout(dense, DROPOUT_STRENGTH)
    dense = Dense(OUTPUT_EMBEDDING_SIZE, activation='relu')(dense)

    if DROPOUT_STRENGTH:
        rnn = Bidirectional(
            GRU(GRU_SIZE,
                return_sequences=True,
                dropout=DROPOUT_STRENGTH,
                unroll=True))(input_seq)
        rnn = Bidirectional(
            GRU(GRU_SIZE,
                return_sequences=False,
                dropout=DROPOUT_STRENGTH,
                unroll=True))(rnn)
    else:
        rnn = Bidirectional(GRU(GRU_SIZE, return_sequences=True,
                                unroll=True))(input_seq)
        rnn = Bidirectional(
            GRU(GRU_SIZE, return_sequences=False, unroll=True))(rnn)

    ensemble = Concatenate()([rnn, dense])

    output = Dense(OUTPUT_EMBEDDING_SIZE, activation='relu')(ensemble)
    output = maybe_dropout(output, DROPOUT_STRENGTH)
    output = Dense(OUTPUT_EMBEDDING_SIZE, activation='sigmoid')(output)

    model = Model(inputs=[input_seq, input_vec], outputs=output)
    model.compile(optimizer="adam", loss='mse')
    return model


def dual_mp(input_shape):
    return _dual_mp(
        input_shape, {
            "embedding_size": 256,
            "min_filter_width": 5,
            "num_filter_widths": 2,
            "filter_count_multiplier": 32,
            "min_filters": 200,
            "conv1d_activation": "relu",
        })


register_model(dual_mp)


def jupyter_model_old(input_shape):
    """Deep  dense"""
    input_seq = Input(
        shape=input_shape['example'], name='example', dtype=K.floatx())

    if len(input_shape['example']) > 1:
        flattened = Flatten()(input_seq)
    else:
        flattened = input_seq

    size = 512
    DROPOUT = 0.1
    #    dense = AlphaDropout(DROPOUT)(flattened)
    dense = flattened
    for i in range(3):
        dense = Dense(
            size,
            activation="selu",
            kernel_initializer="lecun_normal",
            bias_initializer='zeros')(dense)
        dense = Dense(
            size,
            activation="selu",
            kernel_initializer="lecun_normal",
            bias_initializer='zeros')(dense)
        dense = Dense(
            size,
            activation="selu",
            kernel_initializer="lecun_normal",
            bias_initializer='zeros')(dense)

        size /= 2

    dense = Dense(32, activation='sigmoid')(dense)

    model = Model(input_seq, dense)
    model.compile(optimizer="adam", loss='mse')
    return model


register_model(jupyter_model_old)


def visual_model(input_shape):
    """Deep  dense"""
    minipouce_seq = Input(
        shape=input_shape['example'], name='example', dtype=K.floatx())
    visual_input = Input(
        shape=input_shape['vis_example'], name='vis_example', dtype=K.floatx())

    x_combined = Flatten()(minipouce_seq)
    dense = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(x_combined)
    dense = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense)
    dense = Dense(
        128,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense)

    x_vis = Flatten()(visual_input)
    dense2 = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(x_vis)
    dense2 = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense2)
    dense2 = Dense(
        128,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense2)

    with tf.device("/cpu:0"):
        merged = Concatenate()([dense, dense2])
    output = Dense(128)(merged)

    model = Model(inputs=[minipouce_seq, visual_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")

    return model


register_model(visual_model)


def visual_model_decoder(input_shape):
    i = Input(shape=(256, ), name='embedding_input', dtype="float32")

    o = Dense(
        128,
        activation="relu",
        kernel_initializer='random_uniform',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
        name="decode0")(i)
    o = Dense(
        256,
        activation="relu",
        kernel_initializer='random_uniform',
        name="decode2")(o)
    o = Dense(
        1024,
        activation="relu",
        kernel_initializer='random_uniform',
        name="decode4")(o)

    input_size = 1
    for d in input_shape['example']:
        input_size = input_size * d

    vis_input_size = 1
    for d in input_shape['vis_example']:
        vis_input_size = vis_input_size * d

    o1 = Dense(input_size, activation="sigmoid", name="example_out")(o)
    o2 = Dense(vis_input_size, activation="sigmoid", name="vis_example_out")(o)
    m = Model(inputs=i, outputs=[o1, o2])
    m.compile(optimizer="adam", loss="mse")
    return m


register_model(visual_model_decoder)


def decoder(input_shape):
    i = Input(shape=(16, ), name='embedding_input', dtype="float32")
    DROPOUT = 0.1
    dense = AlphaDropout(DROPOUT)(i)
    dense = Dense(
        512,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense)
    dense = AlphaDropout(DROPOUT)(dense)
    dense = Dense(
        1024,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense)
    dense = AlphaDropout(DROPOUT)(dense)

    input_size = 1
    for d in input_shape['example']:
        input_size = input_size * d

    o = Dense(input_size, activation="sigmoid", name="i_out")(dense)
    m = Model(inputs=i, outputs=[o])
    m.compile(optimizer="adam", loss="categorical_crossentropy")
    return m


register_model(decoder)


def iris_decoder(input_shape):
    i = Input(shape=(2, ), name='embedding_input', dtype="float32")
    DROPOUT = 0.1
    dense = AlphaDropout(DROPOUT)(i)
    dense = Dense(
        512,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense)
    dense = AlphaDropout(DROPOUT)(dense)
    dense = Dense(
        1024,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(dense)
    dense = AlphaDropout(DROPOUT)(dense)

    input_size = 1
    for d in input_shape['example']:
        input_size = input_size * d

    o = Dense(input_size, activation="sigmoid", name="i_out")(dense)
    m = Model(inputs=i, outputs=[o])
    m.compile(optimizer="adam", loss="categorical_crossentropy")
    return m


register_model(iris_decoder)


def deep_cnn_w_memory(shape):
    input1 = Input(shape=shape["example"], name='example')

    DROPOUT = .1

    w = Flatten()(input1)
    w = AlphaDropout(DROPOUT)(w)
    w = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(w)
    w = AlphaDropout(DROPOUT)(w)
    w = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(w)
    w = AlphaDropout(DROPOUT)(w)

    x = SeparableConv1D(
        128,
        4,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(input1)
    x = AlphaDropout(DROPOUT)(x)
    x = SeparableConv1D(
        128,
        4,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(x)
    x = AlphaDropout(DROPOUT)(x)
    x = Flatten()(x)

    y = SeparableConv1D(
        128,
        6,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(input1)
    y = AlphaDropout(DROPOUT)(y)
    y = SeparableConv1D(
        128,
        6,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(y)
    y = AlphaDropout(DROPOUT)(y)
    y = Flatten()(y)

    z = SeparableConv1D(
        128,
        2,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(input1)
    z = AlphaDropout(DROPOUT)(z)
    z = SeparableConv1D(
        128,
        2,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(z)
    z = AlphaDropout(DROPOUT)(z)
    z = Flatten()(z)

    d = Concatenate()([w, x, y, z])
    d = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(d)
    d = AlphaDropout(DROPOUT)(d)
    d = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(d)
    d = AlphaDropout(DROPOUT)(d)
    output = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(d)

    model = Model(inputs=input1, outputs=output)

    return model


register_model(deep_cnn_w_memory)


def deep_cnn_2d(shape):
    input1 = Input(shape=shape["example"], name='example')

    DROPOUT = .1

    w = Flatten()(input1)
    w = Dense(
        2048,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(w)
    w = AlphaDropout(DROPOUT)(w)

    y = SeparableConv2D(
        64,
        4,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(input1)
    y = AlphaDropout(DROPOUT)(y)
    y = SeparableConv2D(
        64,
        4,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(y)
    y = AlphaDropout(DROPOUT)(y)
    y = Flatten()(y)

    z = SeparableConv2D(
        64,
        2,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(input1)
    z = AlphaDropout(DROPOUT)(z)
    z = SeparableConv2D(
        64,
        2,
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(z)
    z = AlphaDropout(DROPOUT)(z)
    z = Flatten()(z)

    d = Concatenate()([w, y, z])
    d = Dense(
        256,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(d)
    d = AlphaDropout(DROPOUT)(d)
    d = Dense(
        128,
        activation="selu",
        kernel_initializer="lecun_normal",
        bias_initializer='zeros')(d)
    d = AlphaDropout(DROPOUT)(d)
    output = Dense(
        128,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(d)

    model = Model(inputs=input1, outputs=output)
    #    model.compile(optimizer="adam", loss="mse")
    #    model.summary()
    return model


def mini_mobilenet(shape):
    input1 = Input(shape=shape["example"], name='example')

    DROPOUT = .1

    alpha = 1.0
    depth_multiplier = 1

    x = util._conv_block(input1, 32, alpha, strides=(2, 2))
    x = util._depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = util._depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = util._depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    output = Dense(
        128,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    print("output shape: %s" % output.shape)
    model = Model(inputs=input1, outputs=output)
    #    model.compile(optimizer="adam", loss="mse")
    #    model.summary()
    return model


register_model(mini_mobilenet)


def mobilenet_decoder(input_shape):
    i = Input(shape=(128, ), name='embedding_input', dtype="float32")
    x = Dense(256)(i)
    x = Reshape((16, 16, 1))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    input_size = 1
    for d in input_shape['example']:
        input_size = input_size * d

    o = Dense(
        input_size, activation="sigmoid", name="i_out")(Flatten()(decoded))
    m = Model(inputs=i, outputs=[o])
    m.compile(optimizer="adam", loss="mse")
    return m


register_model(mobilenet_decoder)


def elienet(shape):
    input1 = Input(shape=shape["example"], name='example')
    x = Conv2D(
        32, (3, 3),
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros',
        padding='same')(input1)
    x = Conv2D(
        64, (3, 3),
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros',
        padding='same')(x)
    x = Conv2D(
        128, (3, 3),
        activation='selu',
        kernel_initializer="lecun_normal",
        bias_initializer='zeros',
        padding='same')(x)

    x = MaxPooling2D((3, 3), padding='same')(x)

    x = SeparableConv2D(128, (3, 3), activation='selu', padding='same')(x)
    x = SeparableConv2D(128, (3, 3), activation='selu', padding='same')(x)

    x = Flatten()(x)

    output = Dense(
        256,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    print("Encoder ========")
    return model


register_model(elienet)


def elienet_decoder(input_shape):

    i = Input(shape=(256, ), name='embedding_input', dtype="float32")
    x = Dense(1024)(i)
    x = Reshape((128, 8, 1))(x)

    m = Model(inputs=i, outputs=[x])
    m.compile(optimizer="adam", loss="categorical_crossentropy")
    print("Decoder ========")
    m.summary()
    return m


register_model(elienet_decoder)


def jupyter_model(input_shape):
    """Deep  dense"""
    input_seq = Input(
        shape=input_shape['example'], name='example', dtype=K.floatx())
    x = input_seq

    m = Flatten()(x)
    m = SELU(768)(m)
    m = SELU(512)(m)

    convs = []
    for kernel_size in [1, 2, 3, 4, 5, 6, 7]:
        num_filters = min(200, kernel_size * 50)
        conv = SeparableConv1D(num_filters, kernel_size, **SELU_ARGS)(x)
        conv = Flatten()(conv)
        convs.append(conv)

    x = Concatenate()(convs)
    x = SELU(1024)(x)
    x = Concatenate()([x, m])

    x = SELU(512)(x)
    x = SELU(256)(x)
    x = DenseEmbedding(256)(x)

    model = Model(inputs=input_seq, outputs=x)
    return model


register_model(jupyter_model)


def combo_model(shapes):
    minipouce_input = Input(shape=shapes["example"], name='example')
    rendered_input = Input(shape=shapes["vis_example"], name='vis_example')

    print(K.int_shape(rendered_input))
    print(K.int_shape(minipouce_input))

    jupyter = jupyter_model(shapes)
    elienet = elienet_v2(shapes)

    mp_emb = jupyter(minipouce_input)
    vis_emb = elienet(rendered_input)

    out = DenseEmbedding(256)(Concatenate()([mp_emb, vis_emb]))

    m = Model(inputs=[minipouce_input, rendered_input], outputs=out)
    m.compile(optimizer="adam", loss="mse")
    m.summary()
    return m


register_model(combo_model)


def input_size(shape):
    o = 1
    for i in shape:
        o = o * i
    return o


def combo_model_decoder(shapes):
    i = Input(shape=(256, ), name='embedding_input', dtype="float32")

    print("ex")
    decoded_example = Dense(
        input_size(shapes['example']), activation='sigmoid')(i)
    print(decoded_example)
    decoded_example = Reshape(
        shapes['example'], name='example_out')(decoded_example)
    print(decoded_example)

    print("Vis")
    vis_decoder = elienet_v2_decoder(shapes)
    decoded_vis_example = vis_decoder(i)
    print(decoded_vis_example)

    return Model(inputs=i, outputs=[decoded_example, decoded_vis_example])


register_model(combo_model_decoder)


def jupyter_model(input_shape):
    """Deep  dense"""
    input_seq = Input(
        shape=input_shape['example'], name='example', dtype=K.floatx())
    x = input_seq

    mem = SELU(256)(Flatten()(input_seq))
    mem = SELU(512)(mem)

    xs = []
    for kernel_size in range(2, 5):
        x = SeparableConv1D(
            64, kernel_size, padding="same", **SELU_ARGS)(input_seq)
        xs.append(x)
    x = Concatenate()(xs)

    for depth in range(7):
        for kernel_size in [2, 3]:
            # Compute # filters - input_filters x number of kernel sizes
            conv = SeparableConv1D(
                192,
                kernel_size,
                padding="same",
                kernel_initializer="lecun_normal",
                bias_initializer='zeros')(x)
            x = Add()([x, conv])
            x = Activation("selu")(x)


#            if depth != 7:
#                x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(512, **SELU_ARGS)(x)

    x = SELU(512)(Concatenate()([x, mem]))

    x = Dense(
        256,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = Model(inputs=input_seq, outputs=x)
    return model
register_model(jupyter_model)


def ensemble_model(input_shape):
    """Deep  dense"""

    input_seq = Input(
        shape=input_shape['example'], name='example', dtype=K.floatx())

    # 16x8x8
    x = input_seq

    submodels = []

    dense = SELU(1024)(Flatten()(x))
    dense = SELU(512)(dense)
    dense = SELU(512)(dense)
    submodels.append(dense)

    for kernel_size in [2, 3, 4]:
        c = SeparableConv1D(128, kernel_size, padding="same", **SELU_ARGS)(x)
        c = SeparableConv1D(128, kernel_size, padding="same", **SELU_ARGS)(c)
        c = Flatten()(c)
        submodels.append(c)

    ensemble = Concatenate(axis=1)(submodels)
    ensemble = SELU(256)(ensemble)
    out = Dense(
        256,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(ensemble)

    model = Model(inputs=input_seq, outputs=out)
    return model


def mnist_model(input_shape):

    input_seq = Input(
        shape=input_shape['example'], name='example', dtype=K.floatx())

    o = Conv2D(
        32, kernel_size=(3, 3), activation='relu',
        input_shape=input_shape)(input_seq)
    o = Conv2D(64, (3, 3), activation='relu')(o)
    o = MaxPooling2D(pool_size=(2, 2))(o)

    o = Flatten()(o)
    o = Dense(128, activation='relu')(o)
    o = DenseEmbedding(128)(o)

    model = Model(inputs=input_seq, outputs=o)
    return model


register_model(mnist_model)


def mnist_model_decoder(input_shape):
    i = Input(shape=(128, ), name='embedding_input', dtype="float32")
    o = SELU(512)(i)
    o = SELU(512)(o)

    print(K.int_shape(o))

    target_size = 1
    print(input_shape['example'])
    for d in input_shape['example']:
        target_size = target_size * d

    o = Dense(target_size, activation='sigmoid')(o)
    o = Reshape(input_shape['example'], name='example_out')(o)

    m = Model(inputs=i, outputs=[o])

    return m


register_model(mnist_model_decoder)


def ensemble_model_decoder(input_shape):
    i = Input(shape=(256, ), name='embedding_input', dtype="float32")
    o = SELU(512)(i)
    o = SELU(512)(o)

    print(K.int_shape(o))

    target_size = 1
    print(input_shape['example'])
    for d in input_shape['example']:
        target_size = target_size * d

    o = Dense(target_size, activation='sigmoid')(o)
    o = Reshape(input_shape['example'], name='example_out')(o)

    m = Model(inputs=i, outputs=[o])

    return m


register_model(ensemble_model_decoder)

if __name__ == '__main__':
    input_shape = {"example": [16, 64]}
    m = ensemble_model(input_shape)
    d = ensemble_model_decoder(input_shape)

    m.compile(optimizer="adam", loss="mse")
    m.summary()

    d.compile(optimizer="adam", loss="mse")
    d.summary()

    #    from keras_diagram import ascii
    #    print(ascii(m))

    emb = m.predict([np.zeros([1] + input_shape["example"])])
    dec = d.predict(emb)
