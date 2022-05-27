import os

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


def deserialization_fn(serialized_example):
    fd = {
        "sid": tf.io.FixedLenFeature([], dtype=tf.int64),
        "value": tf.io.FixedLenFeature([], dtype=tf.int64),
    }
    sample = tf.io.parse_single_example(serialized_example, fd)

    return (sample["sid"], sample["value"])


class TFRecordSamplerTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()

        for sid in range(100):
            shard_path = os.path.join(self.get_temp_dir(), f"tfr_{sid}.tfrec")

            with tf.io.TFRecordWriter(str(shard_path)) as w:
                for value in range(1000):
                    example = to_tfrecord(sid, sid * 1000 + value)
                    w.write(example)

    def test_basic(self):
        sampler = TFRecordDatasetSampler(
            self.get_temp_dir(),
            deserialization_fn=deserialization_fn,
            batch_size=10,
            example_per_class=2,
        )

        si = iter(sampler)
        [next(si) for _ in range(10_000)]
        examples = next(si)

        # We should get 3 pairs of shard IDs
        sids = examples[0]
        values = examples[1]
        first_sid = sids[::2]
        second_sid = sids[1::2]

        self.assertLen(sids, 10)
        self.assertAllEqual(first_sid, second_sid)

        for sid, val in zip(sids, values):
            diff = val - sid * 1_000
            self.assertGreaterEqual(diff, 0)
            self.assertLess(diff, 1000)
