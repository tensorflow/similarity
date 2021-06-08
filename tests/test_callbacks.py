import numpy as np
import tensorflow as tf
from tensorflow_similarity.models import SimilarityModel
from tensorflow_similarity.callbacks import EvalCallback
from tensorflow_similarity.callbacks import SplitValidationLoss


def test_eval_callback(tmp_path):
    queries = tf.constant([[1, 2], [1, 2]])
    query_labels = tf.constant([1, 2])
    targets = tf.constant([[1, 2], [1, 2]])
    targets_labels = tf.constant([1, 2])

    log_dir = tmp_path / 'sec/'
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    callback = EvalCallback(queries,
                            query_labels,
                            targets,
                            targets_labels,
                            tb_logdir=str(log_dir))
    # manually set model ^^
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(2, ))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = SimilarityModel(inputs, outputs)
    model.compile('adam', loss='mse', distance='cosine')
    callback.model = model

    # call the only callback method implemented
    callback.on_epoch_end(0, {})


def test_split_val_loss_callback(tmp_path):
    x = tf.constant([[-1]]*5+[[1]]*5)
    y = tf.constant([0]*5+[1]*5)
    known_classes = np.array([1])

    callback = SplitValidationLoss(x, y, known_classes)

    class MockModel:
        def evaluate(self, x, y, verbose=0):
            _, _ = x, verbose
            return float(tf.math.reduce_sum(y) / tf.shape(y)[0])

    callback.model = MockModel()

    # call the only callback method implemented
    logs = {}
    callback.on_epoch_end(0, logs=logs)

    assert logs['known_val_loss'] == 1.0
    assert logs['unknown_val_loss'] == 0.0
