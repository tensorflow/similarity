import tensorflow as tf
from tensorflow_similarity.evaluators import MemoryEvaluator
from tensorflow_similarity.model import CALIBRATION_ACCURACY_TARGETS


def test_config():

    evaluator = MemoryEvaluator(CALIBRATION_ACCURACY_TARGETS)
    config = evaluator.to_config()
    assert "targets" in config
    for k, v in CALIBRATION_ACCURACY_TARGETS.items():
        assert config['targets'][k] == v
    assert config['cutpoints'] == {}
    assert config['thresholds'] == {}

    evaluators2 = MemoryEvaluator().from_config(config)
    assert evaluator.targets == evaluators2.targets
    assert not evaluator.is_calibrated()


def test_config_calibrated():
    evaluator = MemoryEvaluator(CALIBRATION_ACCURACY_TARGETS)
    assert not evaluator.is_calibrated()
    evaluator.cutpoints['optimal'] = 0.1
    evaluator.thresholds['precision'] = [0.2]
    evaluator.thresholds['recall'] = [0.3]

    config = evaluator.to_config()
    assert "targets" in config
    for k, v in CALIBRATION_ACCURACY_TARGETS.items():
        assert config['targets'][k] == v

    assert config['cutpoints']['optimal'] == 0.1
    assert config['thresholds']['recall'] == [0.3]

    evaluators2 = MemoryEvaluator().from_config(config)
    assert evaluator.targets == evaluators2.targets
    assert evaluators2.is_calibrated()
    assert evaluators2.cutpoints['optimal'] == 0.1
    assert evaluators2.thresholds['precision'] == [0.2]


def test_calibration():
    NUM_EXAMPLES = 300
    NUM_LABELS = 3

    # data
    y = tf.random.uniform((NUM_EXAMPLES,), minval=1, maxval=NUM_LABELS)
    labels = tf.random.uniform((NUM_EXAMPLES,), minval=1, maxval=NUM_LABELS)
    distances = tf.random.normal((NUM_EXAMPLES,), mean=0.5, stddev=0.5)

    # calibrate
    ev = MemoryEvaluator()
    calibration = ev.calibrate(y, labels, distances,
                               CALIBRATION_ACCURACY_TARGETS)
    assert ev.is_calibrated()
    assert 'targets' in calibration

    assert 'thresholds' in calibration
    for v in ['precision', 'recall', 'f1_score']:
        assert isinstance(calibration['thresholds'][v], list)

    assert 'cutpoints' in calibration
    assert 'optimal' in calibration['cutpoints']
