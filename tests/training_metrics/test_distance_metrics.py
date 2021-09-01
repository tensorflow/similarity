import tensorflow as tf

from tensorflow_similarity.training_metrics import DistanceMetric
from tensorflow_similarity.training_metrics import DistanceGapMetric
from tensorflow_similarity.distances import CosineDistance
from tensorflow_similarity.types import FloatTensor

EMB1 = tf.nn.l2_normalize([
    [0.5, 1, 0.5],
    [0.2, 0.8, 0.4],
], axis=-1)
EMB2 = tf.nn.l2_normalize([
    [0.66, 0.5, 0.34],
    [0.77, 0.9, 0.92],
], axis=-1)

LABELS = tf.Variable([[1], [1], [2], [2]], dtype='int32')
EMBEDDINGS = tf.Variable(tf.concat((EMB1, EMB2), axis=0))


def cosine(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    distances = 1 - tf.linalg.matmul(a, b, transpose_b=True)
    distances = tf.math.maximum(distances, 0.0)
    return distances


def compute_metric(distance, aggregate, labels, embeddings):
    'Inner function that call the core class'
    metric = DistanceMetric(distance, aggregate=aggregate)
    metric.update_state(labels, embeddings, None)
    return metric.result()


def manual_hard_mining(mydistance, E1, E2):
    l1 = tf.reduce_max(mydistance(E1, E1), axis=1)
    l2 = tf.reduce_max(mydistance(E2, E2), axis=1)
    return [l1, l2]


def test_distance_metric_serialize():
    metric = DistanceMetric('cosine', aggregate='max')
    config = metric.get_config()
    metric2 = DistanceMetric.from_config(config)
    assert isinstance(metric2.distance, CosineDistance)
    assert metric2.aggregate == metric.aggregate

    metric.update_state(LABELS, EMBEDDINGS, None)
    metric2.update_state(LABELS, EMBEDDINGS, None)
    assert metric.result() == metric2.result()


class DistanceMetricsTest(tf.test.TestCase):

    def test_avg_positive(self):

        agg = ['avg', tf.reduce_mean]

        # metric computation
        metric_val = compute_metric('cosine', agg[0], LABELS, EMBEDDINGS)

        # manual computation
        hard_distances = manual_hard_mining(cosine, EMB1, EMB2)
        manual_val = agg[1](hard_distances)
        self.assertAllClose(metric_val, manual_val)

    def test_sum_positive(self):
        agg = ['sum', tf.reduce_sum]

        # metric computation
        metric_val = compute_metric('cosine', agg[0], LABELS, EMBEDDINGS)

        # manual computation
        hard_distances = manual_hard_mining(cosine, EMB1, EMB2)
        manual_val = agg[1](hard_distances)
        self.assertAllClose(metric_val, manual_val)

    def test_max_positive(self):
        agg = ['max', tf.reduce_max]

        # metric computation
        metric_val = compute_metric('cosine', agg[0], LABELS, EMBEDDINGS)

        # manual computation
        hard_distances = manual_hard_mining(cosine, EMB1, EMB2)
        manual_val = agg[1](hard_distances)
        self.assertAllClose(metric_val, manual_val)

    def test_min_positive(self):
        agg = ['min', tf.reduce_min]

        # metric computation
        metric_val = compute_metric('cosine', agg[0], LABELS, EMBEDDINGS)

        # manual computation
        hard_distances = manual_hard_mining(cosine, EMB1, EMB2)
        manual_val = agg[1](hard_distances)
        self.assertAllClose(metric_val, manual_val)


def test_gap():
    metric = DistanceGapMetric('cosine')
    metric.update_state(LABELS, EMBEDDINGS, None)
    metric.result()
