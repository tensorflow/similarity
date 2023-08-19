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
    """"Build an ConvNeXt Model backbone for similarity learning

    [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf)

     Args:
        input_shape: Size of the input image. Must match size of ConvNeXt version you use.
        See below for version input size.

        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.

        variant: Which Variant of the ConvNeXt to use. Defaults to "BASE".

        weights: Use pre-trained weights - the only available currently being
        imagenet. Defaults to "imagenet".

        trainable: Make the ConvNeXt backbone fully trainable or partially
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
    """
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
    """Build the requested ConvNeXt
    
    Args:

        variant: Which Variant of the ConvNeXt to use.
        weights: Use pre-trained weights - the only available currently being
        imagenet.
        trainable: Make the ConvNeXt backbone fully trainable or partially
        trainable.
        - "full" to make the entire backbone trainable,
        - "partial" to only make the last 3 block trainable
        - "frozen" to make it not trainable.
    Returns:
        The output layer of the convnext model
    """
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
