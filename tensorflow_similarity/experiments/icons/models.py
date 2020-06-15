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

import numpy as np
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard, ProgbarLogger
from tensorflow_similarity.experiments.icons.constants import *
from kerastuner.distributions import Fixed
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf


def mobilenet():
    i = Input(shape=(224, 224, 3), dtype=tf.float32, name="image")
    mob = MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=False,
        weights='imagenet',
        pooling=None)
    o = mob(i)
    o = Dense(128)(o)
    m = Model(i, o)
    return m


def resnet():
    i = Input(shape=(48, 48, 3), dtype=tf.float32, name="image")
    base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=i,
        pooling=None)
    o = base(i)
    o = Flatten()(o)
    o = Dense(256, activation="relu")(o)
    o = Dense(128)(o)

    m = Model(i, o)
    return m


def wdsr_model():
    def res_block_b(x_in, num_filters, expansion, kernel_size, linear):
        x = Conv2D(num_filters * expansion, 1, padding='same')(x_in)
        x = Activation('relu')(x)
        x = Conv2D(int(num_filters * linear), 1, padding='same')(x)
        x = Conv2D(num_filters, kernel_size, padding='same')(x)
        x = Add()([x_in, x])
        return x

    def Normalization(rgb_mean=DIV2K_RGB_MEAN, **kwargs):
        return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)

    def wdsr(
            num_filters,
            num_res_blocks,
            res_block_expansion,
            res_block,
            linear):
        x_in = Input(shape=(ICON_SIZE, ICON_SIZE, 3), name="image")
        #x = Normalization()(x_in)

        # main branch (revise padding)
        m = Conv2D(num_filters, 3, padding='same')(x_in)
        for i in range(num_res_blocks):
            m = res_block(m, num_filters, res_block_expansion,
                          kernel_size=3, linear=linear)

        m = Flatten()(m)
        m = Dense(128, activation="sigmoid")(m)

        return Model(x_in, m, name="wdsr-b")

    def wdsr_b(num_filters=32, num_res_blocks=8, res_block_expansion=6):
        linear = Fixed("linear", 0.8)
        return wdsr(
            num_filters,
            num_res_blocks,
            res_block_expansion,
            res_block_b,
            linear=linear)

    return wdsr_b()


def flatten_model():
    i = Input(shape=(ICON_SIZE, ICON_SIZE, 3), name="image")
    o = Flatten()(i)
    dense_size = Fixed("embedding_size", 128)
    o = Dense(dense_size)(o)
    m = Model(i, o)
    return m


def simple_model():
    i = Input(shape=(ICON_SIZE, ICON_SIZE, 3), name="image")
    o = i
    o = Conv2D(64, (3, 3), padding='same', activation='relu')(o)
    o = Conv2D(64, (3, 3), padding='same', activation='relu')(o)
    o = Flatten()(o)
    o = Dense(256, activation="relu")(o)
    o = Dense(256, activation="relu")(o)
    o = Dense(Fixed("embedding_size", SIMILARITY_EMBEDDING_SIZE))(o)
    m = Model(i, o)
    return m
