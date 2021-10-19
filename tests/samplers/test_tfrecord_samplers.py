import numpy as np
import tensorflow as tf

from tensorflow_similarity.samplers import TFRecordDatasetSampler


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tfrecord(sid, value):
    feature = {
        "sid": _int64_feature(sid),
        "value": _int64_feature(value),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def create_data(tmpdir):
    for sid in range(100):
        shard_path = tmpdir / f"tfr_{sid}.tfrec"

        with tf.io.TFRecordWriter(str(shard_path)) as w:
            for value in range(1_000):
                example = to_tfrecord(sid, sid*1_000+value)
                w.write(example)


def deserialization_fn(serialized_example):
    fd = {
       'sid': tf.io.FixedLenFeature([], dtype=tf.int64),
       'value': tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    sample = tf.io.parse_single_example(serialized_example, fd)

    return (sample['sid'], sample['value'])


def test_basic(tmpdir):
    create_data(tmpdir)

    sampler = TFRecordDatasetSampler(
        tmpdir,
        deserialization_fn=deserialization_fn,
        batch_size=10,
        example_per_class=2)

    si = iter(sampler)
    [next(si) for _ in range(10_000)]
    examples = next(si)

    # We should get 3 pairs of shard IDs
    sids = examples[0].numpy()
    values = examples[1].numpy()
    first_sid = sids[::2]
    second_sid = sids[1::2]

    assert len(sids) == 10
    np.testing.assert_array_equal(first_sid, second_sid)
    for sid, val in zip(sids, values):
        assert 0 <= val - sid*1_000 < 1_000
