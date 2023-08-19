# Copyright 2021 The TensorFlow Authors
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
"ConvNeXt backbone for similarity learning"
from __future__ import annotations

import re

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import convnext

from tensorflow_similarity.layers import GeneralizedMeanPooling2D, MetricEmbedding
from tensorflow_similarity.models import SimilarityModel


CONVNEXT_ARCHITECTURE = {
    "TINY": convnext.ConvNeXtTiny,
    "SMALL": convnext.ConvNeXtSmall,
    "BASE": convnext.ConvNeXtBase,
    "LARGE": convnext.ConvNeXtLarge,
    "XLARGE": convnext.ConvNeXtXLarge,
}


def ConvNeXtSim(
    input_shape: tuple[int, int, int],
    embedding_size: int = 128,
    variant: str = "BASE",
    weights: str = "imagenet",
    trainable: str = "frozen",
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = "gem",
    gem_p: float = 3.0,
) -> SimilarityModel:
    inputs = layers.Input(shape=input_shape)
    x = inputs

    if variant not in CONVNEXT_ARCHITECTURE:
        raise ValueError("Unknown ConvNeXt variant. Valid TINY BASE LARGE SMALL XLARGE")

    x = build_convnext(variant, weights, trainable)(x)

    if pooling == "gem":
        x = GeneralizedMeanPooling2D(p=gem_p, name="gem_pool")(x)
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    if include_top and pooling is not None:
        if l2_norm:
            outputs = MetricEmbedding(embedding_size)(x)
        else:
            outputs = layers.Dense(embedding_size)(x)
    else:
        outputs = x

    return SimilarityModel(inputs, outputs)


def build_convnext(variant: str, weights: str | None = None, trainable: str = "full") -> tf.keras.Model:
    convnext_fn = CONVNEXT_ARCHITECTURE[variant.upper()]
    convnext = convnext_fn(weights=weights, include_top=False)

    if trainable == "full":
        convnext.trainable = True
    elif trainable == "partial":
        convnext.trainable = True
        for layer in convnext.layers:
            # freeze all layeres befor the last 3 blocks
            if not re.search("^block[5,6,7]|^top", layer.name):
                layer.trainable = False
    elif trainable == "frozen":
        convnext.trainable = False
    else:
        raise ValueError(f"{trainable} is not a supported option for 'trainable'.")

    if weights:
        for layer in convnext.layers:
            if isinstance(layer, layers.experimental.SyncBatchNormalization):
                layer.trainable = False
    return convnext
