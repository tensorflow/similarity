import re

import pytest
import tensorflow as tf

from tensorflow_similarity.architectures import convnext as convneXt

TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
TF_MINOR_VERSION = int(tf.__version__.split(".")[1])

def tf_version_check(major_version, minor_version):
    if TF_MAJOR_VERSION <= major_version and TF_MINOR_VERSION < minor_version:
        return True

    return False

def test_build_convnext_tiny_full():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = convneXt.build_convnext("tiny", "imagenet", "full")(input_layer)

    convnext = output._keras_history.layer
    assert convnext.name == "convnext_tiny"
    assert convnext.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in convnext._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1
    
    expected_total_layer_count = 151
    expected_trainable_layer_count = 151

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == expected_trainable_layer_count

def test_build_convnext_small_partial():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = convneXt.build_convnext("small", "imagenet", "partial")(input_layer)

    convnext = output._keras_history.layer
    assert convnext.name == "convnext_small"
    assert convnext.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in convnext._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    expected_total_layer_count = 295
    expected_trainable_layer_count = 0

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == expected_trainable_layer_count

def test_build_convnext_base_frozen():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = convneXt.build_convnext("base", "imagenet", "frozen")(input_layer)

    convnext = output._keras_history.layer
    assert convnext.name == "convnext_base"
    assert not convnext.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in convnext._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    expected_total_layer_count = 295
    expected_trainable_layer_count = 0

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == expected_trainable_layer_count

def test_build_convnext_large_full():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = convneXt.build_convnext("large", "imagenet", "full")(input_layer)

    convnext = output._keras_history.layer
    assert convnext.name == "convnext_large"
    assert convnext.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in convnext._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    expected_total_layer_count = 295
    expected_trainable_layer_count = 295

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == expected_trainable_layer_count

