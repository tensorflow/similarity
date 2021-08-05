import re
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import experimental
from tensorflow.keras.layers.experimental import preprocessing
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
def EfficientNetSim(input_shape,
                    embedding_size,
                    variant="B0",
                    weights="imagenet",
                    augmentation="basic",
                    partial_unfreeze=False):

    # input
    inputs = layers.Input(shape=input_shape)
    x = inputs

    if variant not in EFF_INPUT_SIZE:
        raise ValueError("Unknown efficientnet variant. Valid B0...B7")
    img_size = EFF_INPUT_SIZE[variant]

    # augmentation
    if augmentation == "basic":
        # augs usually used in benchmark and work almost always well
        over_size = int(img_size * 1.5)
        augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(over_size, over_size),
            layers.experimental.preprocessing.RandomCrop(img_size, img_size),
            layers.experimental.preprocessing.RandomFlip("horizontal")
        ])

    # add the basic version or the suppplied one.
    if augmentation:
        x = augmentation(x)

    x = build_effnet(x, variant, weights, partial_unfreeze)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(embedding_size)(x)
    return SimilarityModel(inputs, outputs)


def build_effnet(x, variant, weights, partial_unfreeze):

    # init
    effnet_fn = EFF_ARCHITECTURE[variant]
    effnet = effnet_fn(weights=weights, include_top=False)

    # freeze
    if partial_unfreeze:
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
