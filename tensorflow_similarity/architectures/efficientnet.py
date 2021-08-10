"EfficientNet backbone for similarity learning"
import re
from typing import Tuple, Callable, Union
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
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
def EfficientNetSim(input_shape: Tuple[int],
                    embedding_size: int = 128,
                    variant: str = "B0",
                    weights: str = "imagenet",
                    augmentation: Union[Callable, str] = "basic",
                    trainable: str = "frozen"):
    """Build an EffecientNet Model backbone for similarity learning

    Architecture from [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
](https://arxiv.org/abs/1905.11946)

    Args:
        input_shape: Size of the image input prior to augmentation,
        must be bigger than the size of Effnet version you use. See below for
        min input size.

        embedding_size: Size of the output embedding. Usually between 64
        and 512. Defaults to 128.
        variant: Which Variant of the EfficientNEt to use. Defaults to "B0".

        weights: Use pre-trained weights - the only available currently being
        imagenet. Defaults to "imagenet".

        augmentation: How to augment the data - either pass a Sequential model
        of keras.preprocessing.layers or use the built in one or set it to
        None to disable. Defaults to "basic".

        trainable: Make the EfficienNet backbone fully trainable or partially
        trainable. Either "full" to make the entire backbone trainable,
        "partial" to only make the last 3 block trainable or "frozen" to make
        it not trainable. Defaults to "frozen".

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
    img_size = EFF_INPUT_SIZE[variant]

    # augmentation
    # add the basic version or the suppplied one.
    if callable(augmentation):
        x = augmentation(x)
    elif augmentation == "basic":
        # augs usually used in benchmark and work almost always well
        over_size = int(img_size * 1.5)
        x = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(over_size, over_size),
            layers.experimental.preprocessing.RandomCrop(img_size, img_size),
            layers.experimental.preprocessing.RandomFlip("horizontal")
        ])(x)
    else:
        print(f'Warning: {augmentation} is not supported and will be skipped.')

    x = build_effnet(x, variant, weights, trainable)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(embedding_size)(x)
    return SimilarityModel(inputs, outputs)


def build_effnet(x, variant, weights, trainable):
    "Build the requested efficient net."

    # init
    effnet_fn = EFF_ARCHITECTURE[variant]
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
    else:
        effnet.trainable = False

    # wire
    x = efficientnet.preprocess_input(x)
    x = effnet(x)

    return x
