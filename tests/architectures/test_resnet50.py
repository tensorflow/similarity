import re

import pytest
import tensorflow as tf

from tensorflow_similarity.architectures import resnet50


def test_build_full():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = resnet50.build_resnet("imagenet", "full")(input_layer)

    effnet = output._keras_history.layer

    assert effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    assert total_layer_count == 175
    assert trainable_layer_count == 122


def test_build_frozen():
    input_layer = tf.keras.layers.Input((240, 240, 3))
    output = resnet50.build_resnet("imagenet", "frozen")(input_layer)

    effnet = output._keras_history.layer

    assert not effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1

    assert total_layer_count == 175
    assert trainable_layer_count == 0


def test_build_partial():
    input_layer = tf.keras.layers.Input((224, 224, 3))
    output = resnet50.build_resnet("imagenet", "partial")(input_layer)

    effnet = output._keras_history.layer

    assert effnet.trainable

    total_layer_count = 0
    trainable_layer_count = 0
    excluded_layers = 0
    for layer in effnet._self_tracked_trackables:
        total_layer_count += 1
        if layer.trainable:
            trainable_layer_count += 1
            # Check if any of the excluded layers are trainable
            if not re.search("^conv5|^top", layer.name):
                excluded_layers += 1
            if isinstance(layer, tf.keras.layers.experimental.SyncBatchNormalization):
                excluded_layers += 1

    assert total_layer_count == 175
    assert trainable_layer_count == 22
    assert excluded_layers == 0


def test_build_unsupported_trainable():
    msg = "foo is not a supported option for 'trainable'."
    with pytest.raises(ValueError, match=msg):
        _ = resnet50.build_resnet("imagenet", "foo")


def test_include_top():
    input_shape = (224, 224, 3)
    resnet = resnet50.ResNet50Sim(input_shape, include_top=True)

    # The second to last layer should use gem pooling when include_top is True
    assert resnet.layers[-2].name == "gem_pool"
    assert resnet.layers[-2].p == 3.0
    # The default is l2_norm True, so we expect the last layer to be
    # MetricEmbedding.
    assert re.match("metric_embedding", resnet.layers[-1].name) is not None


def test_l2_norm_false():
    input_shape = (224, 224, 3)
    resnet = resnet50.ResNet50Sim(input_shape, include_top=True, l2_norm=False)

    # The second to last layer should use gem pooling when include_top is True
    assert resnet.layers[-2].name == "gem_pool"
    assert resnet.layers[-2].p == 3.0
    # If l2_norm is False, we should return a dense layer as the last layer.
    assert re.match("dense", resnet.layers[-1].name) is not None


@pytest.mark.parametrize(
    "pooling, name", zip(["gem", "avg", "max"], ["gem_pool", "avg_pool", "max_pool"]), ids=["gem", "avg", "max"]
)
def test_include_top_false(pooling, name):
    input_shape = (224, 224, 3)
    resnet = resnet50.ResNet50Sim(input_shape, include_top=False, pooling=pooling)

    # The second to last layer should use gem pooling when include_top is True
    assert resnet.layers[-1].name == name
