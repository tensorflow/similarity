import re

import pytest
import tensorflow as tf

from tensorflow_similarity.architectures import efficientnet


def test_build_effnet_b0_full():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = efficientnet.build_effnet(input_layer, "b0", "imagenet", "full")

    effnet = output._keras_history.layer

    assert effnet.name == "efficientnetb0"
    assert effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    assert total_layer_count == 237
    assert trainable_layer_count == 237


def test_build_effnet_b1_frozen():
    input_layer = tf.keras.layers.Input((240, 240, 3))
    output = efficientnet.build_effnet(input_layer, "b1", "imagenet", "frozen")

    effnet = output._keras_history.layer

    assert effnet.name == "efficientnetb1"
    assert not effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    assert total_layer_count == 339
    assert trainable_layer_count == 0


def test_build_effnet_b0_partial():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = efficientnet.build_effnet(input_layer, "b0", "imagenet", "partial")

    effnet = output._keras_history.layer

    assert effnet.name == "efficientnetb0"
    assert effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    excluded_layers = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1
            # Check if any of the excluded layers are trainable
            if not re.search("^block[5,6,7]|^top", layer.name):
                excluded_layers += 1
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                excluded_layers += 1

    assert total_layer_count == 237
    assert trainable_layer_count == 93
    assert excluded_layers == 0


def test_build_effnet_unsupported_trainable():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    msg = "foo is not a supported option for 'trainable'."
    with pytest.raises(ValueError, match=msg):
        _ = efficientnet.build_effnet(input_layer, "b0", "imagenet", "foo")
