import random
import math

import tensorflow as tf

from tensorflow_similarity.evaluators import MemoryEvaluator
from tensorflow_similarity.classification_metrics import Precision, Recall
from tensorflow_similarity.matchers import MatchNearest
from tensorflow_similarity.types import Lookup


def generate_lookups(num_classes, examples_per_class=10, match_rate=0.7):
    target_labels = []
    lookups = []
    num_match = 0
    total = 0
    for class_id in range(num_classes):
        target_labels.extend([class_id for _ in range(examples_per_class)])

        for _ in range(examples_per_class):
            # draw labels at random
            v = random.uniform(0, 1)
            if v < match_rate:
                label = class_id
                num_match += 1

            else:
                label = random.randint(0, num_classes)
                # case where random pick the same class ^^
                if label == class_id:
                    label += 1
                    label %= num_classes

            e = Lookup(
                    rank=0,
                    distance=round(random.uniform(0.0, 1.0), 3),
                    label=label,
                )
            lookups.append([e])
            total += 1

    effective_match_rate = num_match / total
    return target_labels, lookups, effective_match_rate


def test_evaluate():
    ev = MemoryEvaluator()
    ll = tf.constant([[0], [10], [1], [10], [2], [10], [3], [10]])
    ld = tf.constant([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]])
    res = ev.evaluate_classification(
            query_labels=tf.constant([0, 0, 1, 1, 2, 2, 3, 3]),
            lookup_labels=ll,
            lookup_distances=ld,
            distance_thresholds=tf.constant([math.inf]),
            metrics=[Precision(), Recall()],
            matcher=MatchNearest()
    )

    expected = {
            'precision': 0.5,
            'recall': 1.0,
    }

    for metric_name, val in expected.items():
        assert res[metric_name] == val


def test_comparators():
    # created a regression
    ev = MemoryEvaluator()
    assert ev._is_higher(1,  0)
    assert ev._is_lower(0, 1)

# def test_config():

#     evaluator = MemoryEvaluator(CALIBRATION_ACCURACY_TARGETS)
#     config = evaluator.to_config()
#     assert "targets" in config
#     for k, v in CALIBRATION_ACCURACY_TARGETS.items():
#         assert config['targets'][k] == v
#     assert config['cutpoints'] == {}
#     assert config['thresholds'] == {}

#     evaluators2 = MemoryEvaluator().from_config(config)
#     assert evaluator.targets == evaluators2.targets
#     assert not evaluator.is_calibrated()


# def test_config_calibrated():
#     evaluator = MemoryEvaluator(C7ALIBRATION_ACCURACY_TARGETS)
#     assert not evaluator.is_calibrated()
#     evaluator.cutpoints['optimal'] = 0.1
#     evaluator.thresholds['precision'] = [0.2]
#     evaluator.thresholds['recall'] = [0.3]

#     config = evaluator.to_config()
#     assert "targets" in config
#     for k, v in CALIBRATION_ACCURACY_TARGETS.items():
#         assert config['targets'][k] == v

#     assert config['cutpoints']['optimal'] == 0.1
#     assert config['thresholds']['recall'] == [0.3]

#     evaluators2 = MemoryEvaluator().from_config(config)
#     assert evaluator.targets == evaluators2.targets
#     assert evaluators2.is_calibrated()
#     assert evaluators2.cutpoints['optimal'] == 0.1
#     assert evaluators2.thresholds['precision'] == [0.2]


# def test_calibration():
#     NUM_EXAMPLES = 300
#     NUM_LABELS = 3

#     # data
#     y = tf.random.uniform((NUM_EXAMPLES,), minval=1, maxval=NUM_LABELS)
#     labels = tf.random.uniform((NUM_EXAMPLES,), minval=1, maxval=NUM_LABELS)
#     distances = tf.random.normal((NUM_EXAMPLES,), mean=0.5, stddev=0.5)

#     # calibrate
#     ev = MemoryEvaluator()
#     calibration = ev.calibrate(y, labels, distances,
#                                CALIBRATION_ACCURACY_TARGETS)
#     assert ev.is_calibrated()
#     assert 'targets' in calibration

#     assert 'thresholds' in calibration
#     for v in ['precision', 'recall', 'f1_score']:
#         assert isinstance(calibration['thresholds'][v], list)

#     assert 'cutpoints' in calibration
#     assert 'optimal' in calibration['cutpoints']
