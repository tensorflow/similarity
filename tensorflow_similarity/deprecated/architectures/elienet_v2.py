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

from tensorflow_similarity.architectures.model_registry import register_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow as tf

import os
import pickle
import numpy as np


def sep_conv(x, num_filters, kernel_size=(3, 3), activation='relu'):
    if activation == 'selu':
        x = layers.SeparableConv2D(
            num_filters,
            kernel_size,
            activation='selu',
            padding='same',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros')(x)
    elif activation == 'relu':
        x = layers.SeparableConv2D(
            num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x


def residual(x,
             num_filters,
             kernel_size=(3, 3),
             activation='relu',
             max_pooling=False):
    "Residual block"
    residual = x
    x = sep_conv(x, num_filters, kernel_size, activation)
    x = sep_conv(x, num_filters, kernel_size, activation)
    x = layers.add([x, residual])
    if max_pooling:
        x = layers.MaxPooling2D(kernel_size, strides=(2, 2), padding='same')(x)
    return x


def conv(x, num_filters, kernel_size=(3, 3), activation='relu', strides=(2,
                                                                         2)):
    "2d convolution block"
    if activation == 'selu':
        x = layers.SeparableConv2D(
            num_filters,
            kernel_size,
            strides=strides,
            activation='selu',
            padding='same',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros')(x)
    elif activation == 'relu':
        x = layers.Conv2D(
            num_filters,
            kernel_size,
            strides=strides,
            use_bias=False,
            padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x


def dense(x, dims, activation='relu', batchnorm=True, dropout_rate=0):
    if activation == 'selu':
        x = layers.Dense(
            dims,
            activation='selu',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros')(x)
        if dropout_rate:
            x = layers.AlphaDropout(dropout_rate)(x)
    elif activation == 'relu':
        x = layers.Dense(dims, activation='relu')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x


class ElienetParams:
    def __init__(self,
                 image_shape=(32, 32, 3),
                 kernel_size=(3, 3),
                 enc_expand_start_dims=32,
                 enc_expand_stop_dims=256,
                 enc_num_residual_blocks=4,
                 dense_merge_type='flatten',
                 dense_outer_dims=1024,
                 dense_inner_dims=256,
                 activation='selu',
                 dense_use_bn=True,
                 dec_num_residual_blocks=6):
        self.image_shape = image_shape
        self.image_size = self.image_shape[0] * self.image_shape[
            1] * self.image_shape[2]
        self.channel_size = image_shape[0] * image_shape[1]

        self.kernel_size = kernel_size
        self.enc_expand_start_dims = enc_expand_start_dims
        self.enc_expand_stop_dims = enc_expand_stop_dims
        self.enc_num_residual_blocks = enc_num_residual_blocks
        self.dense_merge_type = dense_merge_type
        self.dense_outer_dims = dense_outer_dims
        self.dense_inner_dims = dense_inner_dims
        self.activation = activation
        self.dense_use_bn = dense_use_bn
        self.dec_num_residual_blocks = dec_num_residual_blocks


def elienet_encoder(hp):
    inputs = layers.Input(shape=hp.image_shape, name="vis_example")
    x = inputs

    # inflate filters
    dims = hp.enc_expand_start_dims
    while dims <= hp.enc_expand_stop_dims:
        x = conv(x, dims, activation=hp.activation)
        x = residual(x, dims, activation=hp.activation)
        dims *= 2

    x = conv(x, dims, activation=hp.activation)

    # residual blocks
    for _ in range(hp.enc_num_residual_blocks):
        x = residual(x, dims, activation=hp.activation)

    if hp.dense_merge_type == 'flatten':
        x = layers.Flatten()(x)
    elif hp.dense_merge_type == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)

    # compress
    dense_dims = hp.dense_outer_dims
    while dense_dims > hp.dense_inner_dims:
        x = dense(
            x, dense_dims, activation=hp.activation, batchnorm=hp.dense_use_bn)
        dense_dims /= 2

    x = layers.Dense(
        hp.dense_inner_dims,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    return Model(inputs, x)


def elienet_decoder(hp):
    inputs = layers.Input(shape=(hp.dense_inner_dims, ), name="embedding")
    x = inputs

    dense_dims = hp.dense_inner_dims * 2

    while dense_dims <= hp.channel_size:

        x = dense(
            x, dense_dims, activation=hp.activation, batchnorm=hp.dense_use_bn)
        dense_dims *= 2

    x = dense(x, hp.channel_size, activation=hp.activation)
    x = layers.Reshape((hp.image_shape[0], hp.image_shape[1], 1))(x)
    for _ in range(hp.dec_num_residual_blocks):
        x = residual(x, hp.image_shape[2], activation=hp.activation)
    # Maybe (1,1) for stride
    decoded = layers.SeparableConv2D(
        hp.image_shape[2], (3, 3), padding='same', activation='sigmoid')(x)
    return Model(inputs, decoded)


def elienet_v2(input_shape):
    ep = ElienetParams(image_shape=input_shape["vis_example"])
    return elienet_encoder(ep)


def elienet_v2_decoder(input_shape):
    ep = ElienetParams(image_shape=input_shape["vis_example"])
    return elienet_decoder(ep)


register_model(elienet_v2)
register_model(elienet_v2_decoder)
