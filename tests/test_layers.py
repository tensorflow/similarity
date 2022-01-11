import math
from unittest import TestCase

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.layers import GeneralizedMeanPooling1D
from tensorflow_similarity.layers import GeneralizedMeanPooling2D


@pytest.fixture
def input_2d_tensor():
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


def test_generalized_mean_pooling_2d(input_2d_tensor):
    result = GeneralizedMeanPooling2D(p=1.0)(input_2d_tensor)
    expected = tf.constant([[5.0, 10.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_2d_inf(input_2d_tensor):
    result = GeneralizedMeanPooling2D(p=math.inf, keepdims=True)(
        input_2d_tensor
    )
    expected = tf.constant([[[[10.0, 15.0]]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_2d_neg_inf(input_2d_tensor):
    result = GeneralizedMeanPooling2D(p=-math.inf, keepdims=True)(
        input_2d_tensor
    )
    expected = tf.constant([[[[0.0, 5.0]]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_2d_zero(input_2d_tensor):
    result = GeneralizedMeanPooling2D(p=0.0)(input_2d_tensor)
    expected = tf.constant([[3.0412402, 8.041241]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_2d_keepdims(input_2d_tensor):
    result = GeneralizedMeanPooling2D(p=1.0, keepdims=True)(input_2d_tensor)
    expected = tf.constant([[[[5.0, 10.0]]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_2d_channels_first(input_2d_tensor):
    result = GeneralizedMeanPooling2D(p=1.0, data_format="channels_first")(
        input_2d_tensor
    )
    # With channels first we expect a (1,3) shape.
    expected = tf.constant([[2.5, 7.5, 12.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_2d_compute_shape():
    input_shape = tf.constant([1, 2, 3, 4])
    result = GeneralizedMeanPooling2D().compute_output_shape(input_shape)
    expected_shape = tf.constant([1, 4])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_2d_compute_shape_keepdims():
    input_shape = tf.constant([1, 2, 3, 4])
    result = GeneralizedMeanPooling2D(keepdims=True).compute_output_shape(
        input_shape
    )
    expected_shape = tf.constant([1, 1, 1, 4])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_2d_compute_shape_dataformat():
    input_shape = tf.constant([1, 2, 3, 4])
    result = GeneralizedMeanPooling2D(
        data_format="channels_first"
    ).compute_output_shape(input_shape)
    expected_shape = tf.constant([1, 2])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_2d_get_config():
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


@pytest.fixture
def input_1d_tensor():
    # A (1, 3, 1) Tensor.
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x = tf.reshape(x, [3, 3, 1])
    return x


def test_generalized_mean_pooling_1d(input_1d_tensor):
    result = GeneralizedMeanPooling1D(p=1.0)(input_1d_tensor)
    expected = tf.constant([[2.0], [5.0], [8.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_1d_inf(input_1d_tensor):
    result = GeneralizedMeanPooling1D(p=math.inf, keepdims=True)(
        input_1d_tensor
    )
    expected = tf.constant(
        [[[3.0], [3.0], [3.0]], [[6.0], [6.0], [6.0]], [[9.0], [9.0], [9.0]]]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_1d_neg_inf(input_1d_tensor):
    result = GeneralizedMeanPooling1D(p=-math.inf, keepdims=True)(
        input_1d_tensor
    )
    expected = tf.constant(
        [[[1.0], [1.0], [1.0]], [[4.0], [4.0], [4.0]], [[7.0], [7.0], [7.0]]]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_1d_zero(input_1d_tensor):
    result = GeneralizedMeanPooling1D(p=0.0)(input_1d_tensor)
    expected = tf.constant([[1.8171206], [4.8171206], [7.8171206]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_1d_keepdims(input_1d_tensor):
    result = GeneralizedMeanPooling1D(p=1.0, keepdims=True)(input_1d_tensor)
    expected = tf.constant([[[2.0]], [[5.0]], [[8.0]]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_1d_channels_first(input_1d_tensor):
    input_1d_tensor = tf.reshape(input_1d_tensor, [3, 1, 3])
    result = GeneralizedMeanPooling1D(p=1.0, data_format="channels_first")(
        input_1d_tensor
    )
    expected = tf.constant([[2.0], [5.0], [8.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-06)


def test_generalized_mean_pooling_1d_compute_shape():
    input_shape = tf.constant([1, 2, 3])
    result = GeneralizedMeanPooling1D().compute_output_shape(input_shape)
    expected_shape = tf.constant([1, 3])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_1d_compute_shape_keepdims():
    input_shape = tf.constant([1, 2, 3])
    result = GeneralizedMeanPooling1D(keepdims=True).compute_output_shape(
        input_shape
    )
    expected_shape = tf.constant([1, 1, 3])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_1d_compute_shape_dataformat():
    input_shape = tf.constant([1, 2, 3])
    result = GeneralizedMeanPooling1D(
        data_format="channels_first"
    ).compute_output_shape(input_shape)
    expected_shape = tf.constant([1, 2])
    np.testing.assert_allclose(result, expected_shape, rtol=1e-06)


def test_generalized_mean_pooling_1d_get_config():
    gem = GeneralizedMeanPooling1D(
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


def test_metric_embedding_get_config():
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
