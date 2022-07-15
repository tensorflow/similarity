import re

import pytest

from tensorflow_similarity.architectures import resnet18


def test_include_top():
    input_shape = (32, 32, 3)
    resnet = resnet18.ResNet18Sim(input_shape, include_top=True)

    # The second to last layer should use gem pooling when include_top is True
    assert resnet.layers[-2].name == "gem_pool"
    assert resnet.layers[-2].p == 3.0
    # The default is l2_norm True, so we expect the last layer to be
    # MetricEmbedding.
    assert re.match("metric_embedding", resnet.layers[-1].name) is not None


def test_l2_norm_false():
    input_shape = (32, 32, 3)
    resnet = resnet18.ResNet18Sim(input_shape, include_top=True, l2_norm=False)

    # The second to last layer should use gem pooling when include_top is True
    assert resnet.layers[-2].name == "gem_pool"
    assert resnet.layers[-2].p == 3.0
    # If l2_norm is False, we should return a dense layer as the last layer.
    assert re.match("dense", resnet.layers[-1].name) is not None


@pytest.mark.parametrize(
    "pooling, name", zip(["gem", "avg", "max"], ["gem_pool", "avg_pool", "max_pool"]), ids=["gem", "avg", "max"]
)
def test_include_top_false(pooling, name):
    input_shape = (32, 32, 3)
    resnet = resnet18.ResNet18Sim(input_shape, include_top=False, pooling=pooling)

    # The second to last layer should use gem pooling when include_top is True
    assert resnet.layers[-1].name == name
