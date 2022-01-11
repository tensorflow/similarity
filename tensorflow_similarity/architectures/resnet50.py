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

"ResNet50 backbone for similarity learning"
import re
from typing import Tuple
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet50
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.layers import GeneralizedMeanPooling2D
from tensorflow_similarity.models import SimilarityModel


# Create an image augmentation pipeline.
def ResNet50Sim(
    input_shape: Tuple[int],
    embedding_size: int = 128,
    weights: str = "imagenet",
    trainable: str = "frozen",
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = "gem",
    gem_p=1.0,
) -> SimilarityModel:
    """Build an ResNet50 Model backbone for similarity learning

    Architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    Args:
        input_shape: Size of the image input prior to augmentation,
        must be bigger than the size of ResNet version you use. See below for
        min input size of 244.

        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.

        weights: Use pre-trained weights - the only available currently being
        imagenet. Defaults to "imagenet".

        trainable: Make the ResNet backbone fully trainable or partially
        trainable.
        - "full" to make the entire backbone trainable,
        - "partial" to only make the last conv5_block trainable
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
    """

    # input
    inputs = layers.Input(shape=input_shape)
    x = inputs

    x = build_resnet(x, weights, trainable)

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


def build_resnet(x: layers.Layer, weights: str, trainable: str) -> layers.Layer:
    """Build the requested ResNet.

    Args:
        x: The input layer to the ResNet.

        weights: Use pre-trained weights - the only available currently being
        imagenet.

        trainable: Make the ResNet backbone fully trainable or partially
        trainable.
        - "full" to make the entire backbone trainable,
        - "partial" to only make the last conv5_block trainable
        - "frozen" to make it not trainable.

    Returns:
        The ouptut layer of the ResNet model
    """

    # init
    resnet = resnet50.ResNet50(weights=weights, include_top=False)

    if trainable == "full":
        resnet.trainable = True
    elif trainable == "partial":
        # let's mark the top part of the network as trainable
        resnet.trainable = True
        for layer in resnet.layers:
            # Freeze all the layers before the the last 3 blocks
            if not re.search("^conv5|^top", layer.name):
                layer.trainable = False
            # don't change the batchnorm weights
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    elif trainable == "frozen":
        resnet.trainable = False
    else:
        raise ValueError(
            f"{trainable} is not a supported option for 'trainable'."
        )

    # wire
    x = resnet(x)

    return x
