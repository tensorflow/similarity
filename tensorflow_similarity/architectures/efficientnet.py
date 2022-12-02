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
"EfficientNet backbone for similarity learning"
from __future__ import annotations

import re

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet

from tensorflow_similarity.layers import GeneralizedMeanPooling2D, MetricEmbedding
from tensorflow_similarity.models import SimilarityModel

from .utils import convert_sync_batchnorm

EFF_INPUT_SIZE = {
    "B0": 224,
    "B1": 240,
    "B2": 260,
    "B3": 300,
    "B4": 380,
    "B5": 456,
    "B6": 528,
    "B7": 600,
}

EFF_ARCHITECTURE = {
    "B0": efficientnet.EfficientNetB0,
    "B1": efficientnet.EfficientNetB1,
    "B2": efficientnet.EfficientNetB2,
    "B3": efficientnet.EfficientNetB3,
    "B4": efficientnet.EfficientNetB4,
    "B5": efficientnet.EfficientNetB5,
    "B6": efficientnet.EfficientNetB6,
    "B7": efficientnet.EfficientNetB7,
}


def EfficientNetSim(
    input_shape: tuple[int, int, int],
    embedding_size: int = 128,
    variant: str = "B0",
    weights: str = "imagenet",
    trainable: str = "frozen",
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = "gem",
    gem_p: float = 3.0,
) -> SimilarityModel:
    """Build an EfficientNet Model backbone for similarity learning

    [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

    Args:
        input_shape: Size of the input image. Must match size of EfficientNet version you use.
        See below for version input size.

        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.

        variant: Which Variant of the EfficientNet to use. Defaults to "B0".

        weights: Use pre-trained weights - the only available currently being
        imagenet. Defaults to "imagenet".

        trainable: Make the EfficientNet backbone fully trainable or partially
        trainable.
        - "full" to make the entire backbone trainable,
        - "partial" to only make the last 3 block trainable
        - "frozen" to make it not trainable.

        l2_norm: If True and include_top is also True, then
        tfsim.layers.MetricEmbedding is used as the last layer, otherwise
        keras.layers.Dense is used. This should be true when using cosine
        distance. Defaults to True.

        include_top: Whether to include the fully-connected layer at the top
        of the network. Defaults to True.

        pooling: Optional pooling mode for feature extraction when
        include_top is False. Defaults to gem.
        - None means that the output of the model will be the 4D tensor
          output of the last convolutional layer.
        - avg means that global average pooling will be applied to the
          output of the last convolutional layer, and thus the output of the
          model will be a 2D tensor.
        - max means that global max pooling will be applied.
        - gem means that global GeneralizedMeanPooling2D will be applied.
          The gem_p param sets the contrast amount on the pooling.

        gem_p: Sets the power in the GeneralizedMeanPooling2D layer. A value
        of 1.0 is equivalent to GlobalMeanPooling2D, while larger values
        will increase the contrast between activations within each feature
        map, and a value of math.inf will be equivalent to MaxPool2d.

    Note:
        EfficientNet expects images at the following size:
         - "B0": 224,
         - "B1": 240,
         - "B2": 260,
         - "B3": 300,
         - "B4": 380,
         - "B5": 456,
         - "B6": 528,
         - "B7": 600,

    """

    # input
    inputs = layers.Input(shape=input_shape)
    x = inputs

    if variant not in EFF_INPUT_SIZE:
        raise ValueError("Unknown efficientnet variant. Valid B0...B7")

    x = build_effnet(variant, weights, trainable)(x)

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


def build_effnet(variant: str, weights: str | None = None, trainable: str = "full") -> tf.keras.Model:
    """Build the requested efficient net.

    Args:

        variant: Which Variant of the EfficientNet to use.

        weights: Use pre-trained weights - the only available currently being
        imagenet.

        trainable: Make the EfficientNet backbone fully trainable or partially
        trainable.
        - "full" to make the entire backbone trainable,
        - "partial" to only make the last 3 block trainable
        - "frozen" to make it not trainable.

    Returns:
        The output layer of the efficientnet model
    """

    # init
    effnet_fn = EFF_ARCHITECTURE[variant.upper()]
    effnet = effnet_fn(weights=weights, include_top=False)
    effnet = convert_sync_batchnorm(effnet)

    if trainable == "full":
        effnet.trainable = True
    elif trainable == "partial":
        # let's mark the top part of the network as trainable
        effnet.trainable = True
        for layer in effnet.layers:
            # Freeze all the layers before the the last 3 blocks
            if not re.search("^block[5,6,7]|^top", layer.name):
                layer.trainable = False
    elif trainable == "frozen":
        effnet.trainable = False
    else:
        raise ValueError(f"{trainable} is not a supported option for 'trainable'.")

    # Don't train the BN layers if we are loading pre-trained weights.
    if weights:
        for layer in effnet.layers:
            if isinstance(layer, layers.experimental.SyncBatchNormalization):
                layer.trainable = False

    return effnet
