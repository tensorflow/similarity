import re

import pytest
import tensorflow as tf

from tensorflow_similarity.architectures import efficientnet

# TODO(ovallis): rewrite these tests so they aren't so brittle.
TF_MAJOR_VERSION = int(tf.__version__.split(".")[0])
TF_MINOR_VERSION = int(tf.__version__.split(".")[1])


def tf_version_check(major_version, minor_version):
    if TF_MAJOR_VERSION <= major_version and TF_MINOR_VERSION < minor_version:
        return True

    return False


def test_build_effnet_b0_full():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = efficientnet.build_effnet("b0", "imagenet", "full")(input_layer)

    effnet = output._keras_history.layer

    assert effnet.name == "efficientnetb0"
    assert effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    if tf_version_check(2, 9):
        expected_total_layer_count = 237
        expected_trainable_layer_count = 188
    else:
        expected_total_layer_count = 238
        expected_trainable_layer_count = 189

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == expected_trainable_layer_count


def test_build_effnet_b1_frozen():
    input_layer = tf.keras.layers.Input((240, 240, 3))
    output = efficientnet.build_effnet("b1", "imagenet", "frozen")(input_layer)

    effnet = output._keras_history.layer

    assert effnet.name == "efficientnetb1"
    assert not effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    if tf_version_check(2, 9):
        expected_total_layer_count = 339
    else:
        expected_total_layer_count = 340

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == 0


def test_build_effnet_b0_partial():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = efficientnet.build_effnet("b0", "imagenet", "partial")(input_layer)

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

    expected_trainable_layer_count = 93

    if tf_version_check(2, 9):
        expected_total_layer_count = 237
    else:
        expected_total_layer_count = 238

    assert total_layer_count == expected_total_layer_count
    assert trainable_layer_count == expected_trainable_layer_count
    assert excluded_layers == 0


def test_build_effnet_unsupported_trainable():
    msg = "foo is not a supported option for 'trainable'."
    with pytest.raises(ValueError, match=msg):
        _ = efficientnet.build_effnet("b0", "imagenet", "foo")


def test_unsuported_varient():
    input_shape = (224, 224, 3)
    msg = "Unknown efficientnet variant. Valid B0...B7"
    with pytest.raises(ValueError, match=msg):
        _ = efficientnet.EfficientNetSim(input_shape, 128, "bad_varient")


def test_include_top():
    input_shape = (224, 224, 3)
    effnet = efficientnet.EfficientNetSim(input_shape, include_top=True)

    # The second to last layer should use gem pooling when include_top is True
    assert effnet.layers[-2].name == "gem_pool"
    assert effnet.layers[-2].p == 3.0
    # The default is l2_norm True, so we expect the last layer to be
    # MetricEmbedding.
    assert re.match("metric_embedding", effnet.layers[-1].name) is not None


def test_l2_norm_false():
    input_shape = (224, 224, 3)
    effnet = efficientnet.EfficientNetSim(input_shape, include_top=True, l2_norm=False)

    # The second to last layer should use gem pooling when include_top is True
    assert effnet.layers[-2].name == "gem_pool"
    assert effnet.layers[-2].p == 3.0
    # If l2_norm is False, we should return a dense layer as the last layer.
    assert re.match("dense", effnet.layers[-1].name) is not None


@pytest.mark.parametrize(
    "pooling, name",
    zip(["gem", "avg", "max"], ["gem_pool", "avg_pool", "max_pool"]),
    ids=["gem", "avg", "max"],
)
def test_include_top_false(pooling, name):
    input_shape = (224, 224, 3)
    effnet = efficientnet.EfficientNetSim(input_shape, include_top=False, pooling=pooling)

    # The second to last layer should use gem pooling when include_top is True
    assert effnet.layers[-1].name == name
