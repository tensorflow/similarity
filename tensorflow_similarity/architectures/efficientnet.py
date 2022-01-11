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
import re
from typing import Tuple
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.layers import GeneralizedMeanPooling2D
from tensorflow_similarity.models import SimilarityModel

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


# Create an image augmentation pipeline.
def EfficientNetSim(
    input_shape: Tuple[int],
    embedding_size: int = 128,
    variant: str = "B0",
    weights: str = "imagenet",
    trainable: str = "frozen",
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = "gem",
    gem_p=1.0,
) -> SimilarityModel:
    """Build an EffecientNet Model backbone for similarity learning

    Architecture from [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

    Args:
        input_shape: Size of the image input prior to augmentation,
        must be bigger than the size of Effnet version you use. See below for
        min input size.

        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.

        variant: Which Variant of the EfficientNet to use. Defaults to "B0".

        weights: Use pre-trained weights - the only available currently being
        imagenet. Defaults to "imagenet".

        trainable: Make the EfficienNet backbone fully trainable or partially
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
        of 1.0 is equivelent to GlobalMeanPooling2D, while larger values
        will increase the contrast between activations within each feature
        map, and a value of math.inf will be equivelent to MaxPool2d.

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

    x = build_effnet(x, variant, weights, trainable)

    if include_top:
        x = GeneralizedMeanPooling2D(p=gem_p, name="gem_pool")(x)
        if l2_norm:
            outputs = MetricEmbedding(embedding_size)(x)
        else:
            outputs = layers.Dense(embedding_size)(x)
    else:
        if pooling == "gem":
            x = GeneralizedMeanPooling2D(p=gem_p, name="gem_pool")(x)
        elif pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        outputs = x

    return SimilarityModel(inputs, outputs)


def build_effnet(
    x: layers.Layer, variant: str, weights: str, trainable: str
) -> layers.Layer:
    """Build the requested efficient net.

    Args:
        x: The input layer to the efficientnet.

        variant: Which Variant of the EfficientNet to use.

        weights: Use pre-trained weights - the only available currently being
        imagenet.

        trainable: Make the EfficienNet backbone fully trainable or partially
        trainable.
        - "full" to make the entire backbone trainable,
        - "partial" to only make the last 3 block trainable
        - "frozen" to make it not trainable.

    Returns:
        The ouptut layer of the efficientnet model
    """

    # init
    effnet_fn = EFF_ARCHITECTURE[variant.upper()]
    effnet = effnet_fn(weights=weights, include_top=False)

    if trainable == "full":
        effnet.trainable = True
    elif trainable == "partial":
        # let's mark the top part of the network as trainable
        effnet.trainable = True
        for layer in effnet.layers:
            # Freeze all the layers before the the last 3 blocks
            if not re.search("^block[5,6,7]|^top", layer.name):
                layer.trainable = False
            # don't change the batchnorm weights
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    elif trainable == "frozen":
        effnet.trainable = False
    else:
        raise ValueError(
            f"{trainable} is not a supported option for 'trainable'."
        )

    # wire
    x = effnet(x)

    return x
