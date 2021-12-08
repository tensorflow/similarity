from unittest import TestCase

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_similarity.schedules import WarmUpCosine

testdata = [
    range(10),
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9],
    np.arange(10),
    tf.range(10, dtype="float32"),
]

test_ids = ["python range", "python list", "numpy arange", "tf range"]


@pytest.mark.parametrize("steps", testdata, ids=test_ids)
def test_warmup_cosine(steps):
    warmup_cosine = WarmUpCosine(
        initial_learning_rate=1.0,
        decay_steps=10,
        warmup_steps=2,
        warmup_learning_rate=0.5,
        alpha=0.1,
        name="foo",
    )

    lrs = warmup_cosine(steps)

    expected_lrs = tf.constant(
        [
            0.5,
            0.7334816,
            0.9140576,
            0.8145034,
            0.68905765,
            0.54999995,
            0.41094226,
            0.28549662,
            0.18594232,
            0.12202458,
        ]
    )

    np.testing.assert_allclose(expected_lrs, lrs, rtol=1e-06)


def test_warmup_cosine_no_warmup():
    warmup_cosine = WarmUpCosine(
        initial_learning_rate=1.0,
        decay_steps=10,
        warmup_steps=0,
        warmup_learning_rate=0.5,
        alpha=0.1,
        name="foo",
    )

    lrs = warmup_cosine(range(10))

    expected_lrs = tf.constant(
        [
            1.0,
            0.977975,
            0.9140576,
            0.8145034,
            0.68905765,
            0.54999995,
            0.41094226,
            0.28549662,
            0.18594232,
            0.12202458,
        ]
    )

    np.testing.assert_allclose(expected_lrs, lrs, rtol=1e-06)


def test_warmup_cosine_config():
    warmup_cosine = WarmUpCosine(
        initial_learning_rate=1.0,
        decay_steps=10,
        warmup_steps=2,
        warmup_learning_rate=0.0,
        alpha=0.1,
        name="foo",
    )

    config = warmup_cosine.get_config()

    expected_config = {
        "initial_learning_rate": 1.0,
        "decay_steps": 10,
        "alpha": 0.1,
        "warmup_learning_rate": 0.0,
        "warmup_steps": 2,
        "name": "foo",
    }

    TestCase().assertDictEqual(expected_config, config)


def test_warmup_cosine_assert_smaller_warmup_learning_rate():
    msg = "warmup_learning_rate must be smaller than the initial_learning_rate"
    with pytest.raises(ValueError, match=msg):
        _ = WarmUpCosine(
            initial_learning_rate=1.0,
            decay_steps=10,
            warmup_steps=5,
            warmup_learning_rate=1.5,
        )


def test_warmup_cosine_assert_smaller_warmup_steps():
    msg = "warmup_steps must be smaller than the decay_steps"
    with pytest.raises(ValueError, match=msg):
        _ = WarmUpCosine(
            initial_learning_rate=1.0,
            decay_steps=10,
            warmup_steps=20,
        )
