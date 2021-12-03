import math
from unittest import TestCase

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.layers import GeneralizedMeanPooling2D


@pytest.fixture
def input_tensor():
    # A (1, 3, 3, 2) Tensor.
    return tf.constant(
        [
            [
                [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]],
                [[5.0, 10.0], [5.0, 10.0], [5.0, 10.0]],
                [[10.0, 15.0], [10.0, 15.0], [10.0, 15.0]],
            ],
        ]
    )


def test_generalized_mean_pooling(input_tensor):
    result = GeneralizedMeanPooling2D(p=1.0)(input_tensor)
    expected = tf.constant([[5.0, 10.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_inf(input_tensor):
    result = GeneralizedMeanPooling2D(p=math.inf, keepdims=True)(input_tensor)
    expected = tf.constant([[[[10.0, 15.0]]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_neg_inf(input_tensor):
    result = GeneralizedMeanPooling2D(p=-math.inf, keepdims=True)(input_tensor)
    expected = tf.constant([[[[0.0, 5.0]]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_zero(input_tensor):
    result = GeneralizedMeanPooling2D(p=0.0)(input_tensor)
    expected = tf.constant([[3.0412402, 8.041241]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_keepdims(input_tensor):
    result = GeneralizedMeanPooling2D(p=1.0, keepdims=True)(input_tensor)
    expected = tf.constant([[[[5.0, 10.0]]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_channels_first(input_tensor):
    result = GeneralizedMeanPooling2D(p=1.0, data_format="channels_first")(
        input_tensor
    )
    # With channels first we expect a (1,3) shape.
    expected = tf.constant([[2.5, 7.5, 12.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_compute_shape():
    input_shape = tf.constant([1, 2, 3, 4])
    result = GeneralizedMeanPooling2D().compute_output_shape(input_shape)
    expected_shape = tf.constant([1, 4])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_compute_shape_keepdims():
    input_shape = tf.constant([1, 2, 3, 4])
    result = GeneralizedMeanPooling2D(keepdims=True).compute_output_shape(
        input_shape
    )
    expected_shape = tf.constant([1, 1, 1, 4])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_compute_shape_dataformat():
    input_shape = tf.constant([1, 2, 3, 4])
    result = GeneralizedMeanPooling2D(
        data_format="channels_first"
    ).compute_output_shape(input_shape)
    expected_shape = tf.constant([1, 2])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_get_config():
    gem = GeneralizedMeanPooling2D(
        p=3.0, data_format="channels_first", keepdims=True, name="GEM"
    )
    config = gem.get_config()
    expected_config = {
        "p": 3.0,
        "data_format": "channels_first",
        "keepdims": True,
        "name": "GEM",
        "trainable": True,
        "dtype": "float32",
    }
    TestCase().assertDictEqual(expected_config, config)


def test_metric_embedding():
    input_tensor = tf.constant([[4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    me_layer = MetricEmbedding(
        4, kernel_initializer=tf.constant_initializer(1.0)
    )
    result = me_layer(input_tensor)
    expected_result = tf.constant([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(result, expected_result, rtol=1e-06)


def test_generalized_mean_pooling_get_config():
    me_layer = MetricEmbedding(32)
    config = me_layer.get_config()
    expected_config = {
        "activation": None,
        "activity_regularizer": None,
        "bias_constraint": None,
        "bias_initializer": "zeros",
        "bias_regularizer": None,
        "dtype": "float32",
        "kernel_constraint": None,
        "kernel_initializer": "glorot_uniform",
        "kernel_regularizer": None,
        "name": "metric_embedding_1",
        "trainable": True,
        "units": 32,
        "use_bias": True,
    }
    TestCase().assertDictEqual(expected_config, config)
