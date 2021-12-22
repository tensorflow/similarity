import math

import pytest
import tensorflow as tf

from tensorflow_similarity import utils
from tensorflow_similarity import types


@pytest.fixture
def lookups():
    lookups = []
    for i in range(2):
        lookup_set = []
        for j in range(2):
            label = j + i * 2
            lookup_set.append(
                types.Lookup(rank=j, distance=float(j), label=label))

        lookups.append(lookup_set)

    return lookups


def test_is_tensor():
    assert utils.is_tensor_or_variable(tf.constant([0]))


def test_is_variable():
    assert utils.is_tensor_or_variable(tf.Variable(1.0))


def test_is_not_tensor_or_variable():
    assert not utils.is_tensor_or_variable([0])


def test_unpack_lookup_labels(capsys, lookups):
    unpacked = utils.unpack_lookup_labels(lookups)
    expected = tf.constant([[0, 1], [2, 3]], dtype="int32")

    assert tf.reduce_all(tf.math.equal(unpacked, expected))

    captured = capsys.readouterr()
    assert captured.out == ""


def test_unpack_lookup_labels_uneven_lookup_sets(capsys, lookups):
    # Add an extra label to the second lookup set
    lookups[1].append(types.Lookup(rank=3, distance=math.inf, label=4))

    unpacked = utils.unpack_lookup_labels(lookups)
    expected = tf.constant([[0, 1, 0x7FFFFFFF], [2, 3, 4]], dtype="int32")

    assert tf.reduce_all(tf.math.equal(unpacked, expected))

    captured = capsys.readouterr()
    msg = ("WARNING: 1 lookup sets are shorter than the max lookup set "
           "length. Imputing 0x7FFFFFFF for the missing label lookups.\n")

    assert captured.out == msg


def test_unpack_lookup_distances(capsys, lookups):
    unpacked = utils.unpack_lookup_distances(lookups)
    expected = tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype="float32")

    assert tf.reduce_all(tf.math.equal(unpacked, expected))

    captured = capsys.readouterr()
    assert captured.out == ""


def test_unpack_lookup_distances_uneven_lookup_sets(capsys, lookups):
    # Add an extra label to the second lookup set
    lookups[1].append(types.Lookup(rank=3, distance=2.0, label=4))

    unpacked = utils.unpack_lookup_distances(lookups)
    expected = tf.constant([[0.0, 1.0, math.inf], [0.0, 1.0, 2.0]],
                           dtype="float32")

    assert tf.reduce_all(tf.math.equal(unpacked, expected))

    captured = capsys.readouterr()
    msg = ("WARNING: 1 lookup sets are shorter than the max lookup set "
           "length. Imputing math.inf for the missing distance lookups.\n")

    assert captured.out == msg


def test_same_length_rows_check_same_length():
    x = tf.ragged.constant([[0, 1], [0, 2], [0, 3]])
    assert utils._same_length_rows(x)


def test_same_length_rows_check_different_lengths():
    x = tf.ragged.constant([[0], [0, 2], [0, 2, 3]])
    assert not utils._same_length_rows(x)


def test_count_of_small_lookup_sets_ragged():
    x = tf.ragged.constant([[0], [0, 2], [0, 2, 3]])
    counts = utils._count_of_small_lookup_sets(x)
    assert counts == 2


def test_count_of_small_lookup_sets_all_same_length():
    x = tf.ragged.constant([[0, 2], [0, 2], [0, 2]])
    counts = utils._count_of_small_lookup_sets(x)
    assert counts == 0
