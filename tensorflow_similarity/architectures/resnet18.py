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
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import imagenet_utils
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.layers import GeneralizedMeanPooling2D
from tensorflow_similarity.models import SimilarityModel


# Create an image augmentation pipeline.
def ResNet18Sim(
    input_shape: Tuple[int, int, int],
    embedding_size: int = 128,
    l2_norm: bool = True,
    include_top: bool = True,
    pooling: str = "gem",
    gem_p=1.0,
    preproc_mode: str = "torch",
    similarity_model: bool = True,
) -> SimilarityModel:
    """Build an ResNet18 Model backbone for similarity learning

    Architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    Args:
        input_shape: Expected to be betweeen 32 and 224 and in the (H, W, C)
        data_format augmentation function.

        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.

        l2_norm: If True and include_top is also True, then
        tfsim.layers.MetricEmbedding is used as the last layer, otherwise
        keras.layers.Dense is used. This should be true when using cosine
        distance. Defaults to True.

        include_top: Whether to include a fully-connected layer of
        embedding_size at the top of the network. Defaults to True.

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

        preproc_mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR, then will zero-center
          each color channel with respect to the ImageNet dataset, without
          scaling.
        - tf: will scale pixels between -1 and 1, sample-wise.
        - torch: will scale pixels between 0 and 1 and then will normalize each
          channel with respect to the ImageNet dataset.
    """

    # input
    inputs = layers.Input(shape=input_shape)
    x = inputs

    resnet = build_resnet(x, "channels_last", preproc_mode)
    x = resnet(x)

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

    return SimilarityModel(inputs, outputs, name="resnet18sim")


def build_resnet(x: layers.Layer, data_format, preproc_mode) -> layers.Layer:
    """Build the requested ResNet.

    Args:
        x: The input layer to the ResNet.

        data_format: Data format of the image tensor.

        preproc_mode: One of "caffe", "tf" or "torch".

    Returns:
        The ouptut layer of the ResNet model
    """
    # Handle the case where x.shape includes the placeholder for the batch dim.
    if x.shape[0] is None:
        inputs = layers.Input(shape=x.shape[1:])
    else:
        inputs = layers.Input(shape=x.shape)

    x = imagenet_utils.preprocess_input(
        x, data_format=data_format, mode=preproc_mode
    )
    x = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name="conv1_pad"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        strides=1,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name="conv1_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name="conv1_bn")(x)
    x = tf.keras.layers.Activation("relu", name="conv1_relu")(x)

    outputs = stack_fn(x)

    model = tf.keras.Model(inputs, outputs, name="resnet18")
    return model


def block0(
    x,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    name=None,
):
    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.LecunUniform(),
            name=name + "_0_conv",
        )(x)
        shortcut = tf.keras.layers.BatchNormalization(
            epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="SAME",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name=name + "_1_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_1_bn"
    )(x)
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.LecunUniform(),
        name=name + "_2_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1.001e-5, name=name + "_2_bn"
    )(x)

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.Activation("relu", name=name + "_out")(x)
    return x


def stack0(x, filters, blocks, stride1=2, name=None):
    x = block0(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block0(
            x,
            filters,
            conv_shortcut=False,
            name=name + "_block" + str(i),
        )
    return x


def stack_fn(x):
    x = stack0(x, 64, 2, stride1=1, name="conv2")
    x = stack0(x, 128, 2, name="conv3")
    x = stack0(x, 256, 2, name="conv4")
    return stack0(x, 512, 2, name="conv5")
